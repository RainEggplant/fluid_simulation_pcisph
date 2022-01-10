import taichi as ti
from particle_system import ParticleSystem
from pcisph import PCISPHSolver

ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)


if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    ps.add_cube(lower_corner=[5.5, 1],
                cube_size=[3.0, 3.0],
                velocity=[0.0, -10.0],
                density=1000.0,
                color=0x0ea144,
                material=1)

    ps.add_cube(lower_corner=[2, 3],
                cube_size=[3.0, 3.0],
                velocity=[0.0, -10.0],
                density=1000.0,
                color=0xeb4034,
                material=1)

    ps.add_cube(lower_corner=[5, 7],
                cube_size=[1.0, 1.0],
                velocity=[0.0, -10.0],
                density=1000.0,
                color=0x175cfc,
                material=1)

    pcisph_solver = PCISPHSolver(ps)
    pcisph_solver.dt[None] = 1e-4

    gui = ti.GUI(background_color=0xa2b7bd)
    while gui.running:
        for i in range(10):
            pcisph_solver.step()

        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=particle_info['color'])
        gui.show()
