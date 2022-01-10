import taichi as ti
from sph_base import SPHBase


class PCISPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.density_err = ti.field(float, shape=())
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)

        self.k_pci = ti.field(float, shape=())

    @ti.kernel
    def compute_k_pci(self):
        """
        compute
            β = Δt^2 m^2 * 2 / ρ_0^2
            k_pci = 1 / (β * (sum_j ∇W_{ij} · sum_j ∇W_{ij} + sum_j(∇W_{ij} · ∇W_{ij})))
        """
        beta = 2 * (self.dt[None] * self.ps.m_V) ** 2

        # perfect sampling
        half_n_particles = ti.cast(self.ps.support_radius / self.ps.particle_radius, int)
        grad_sum = ti.Vector([0.0 for _ in range(self.ps.dim)])
        grad_dot_sum = 0.0
        for offset in ti.grouped(ti.ndrange(*((-half_n_particles, half_n_particles + 1),) * self.ps.dim)):
            r = offset * self.ps.particle_radius
            grad = self.cubic_kernel_derivative(r)
            grad_sum += grad
            grad_dot_sum += grad.dot(grad)

        self.k_pci[None] = -1.0 / ti.max(beta * (grad_sum.dot(grad_sum) + grad_dot_sum), 1e-6)

    @ti.kernel
    def compute_densities(self):
        self.density_err[None] = 0.
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] *= self.density_0
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)  # avoid negative pressure
            density_err = (self.density_0 - self.ps.density[p_i])
            self.density_err[None] += ti.abs(density_err)
        self.density_err[None] /= self.ps.particle_num[None]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            # Add body force
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[self.ps.dim-1] = self.g
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    @ti.kernel
    def prepare_iteration(self):
        # compute predicted pressure
        for p_i in range(self.ps.particle_num[None]):
            self.ps.pressure[p_i] = self.k_pci[None] * (self.density_0 - self.ps.density[p_i])

        # compute pressure force and acceleration
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.pressure_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def pressure_iteration(self):
        self.density_err[None] = 0.
        # compute refined pressure
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            density_p = 0.
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                density_p += (self.d_velocity[p_i] - self.d_velocity[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
            density_p *= self.ps.m_V * self.density_0 * (self.dt[None] ** 2)
            self.density_err[None] += ti.abs(self.density_0 - self.ps.density[p_i] + density_p)
            self.ps.pressure[p_i] += self.k_pci[None] * (self.density_0 - self.ps.density[p_i] - density_p)
        self.density_err[None] /= self.ps.particle_num[None]

        # compute pressure force and acceleration
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.pressure_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def advect_pressure(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.dt[None] * self.d_velocity[p_i]

    def substep(self):
        self.compute_non_pressure_forces()
        self.advect()
        self.compute_k_pci()

        i = 0
        self.compute_densities()
        self.prepare_iteration()
        while self.density_err[None] / self.density_0 > 1e-3 and i < 1000:
            i += 1
        self.advect_pressure()
