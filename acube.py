import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# Parameters for mpm
quality = 1
size = 64
n_particles, n_grid = size**2 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho

E, nu = 5e3, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))

x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)

Jp = ti.field(dtype=float, shape=n_particles)
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
gravity = ti.Vector.field(2, dtype=float, shape=())

# Parameters for levels
levels = 4
now_level = ti.field(dtype=int, shape=())
now_level[None] = 0
ifrun = True
num_walls = 5
num_goals = 5
walls = ti.Matrix.field(2,2,dtype=int,shape=(levels, num_walls))
knife = ti.Matrix.field(2,2,dtype=float,shape=levels)
pos_start = ti.Vector.field(2,dtype=float, shape = levels)
pos_end = ti.Vector.field(2,dtype=int, shape = (levels, num_goals))

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30  # gravity

            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
            # Collision detection
            for n_wall in range(num_walls):       
                if i >= walls[now_level[None], n_wall][0,0] and i <= walls[now_level[None], n_wall][1,0] and j >= walls[now_level[None],n_wall][1,1] and j <= walls[now_level[None],n_wall][0,1]:
                    if i == walls[now_level[None],n_wall][0,0]:
                        grid_v[i, j][0] = 0
                    if i == walls[now_level[None],n_wall][1,0]:
                        grid_v[i, j][0] = 0
                    if j == walls[now_level[None],n_wall][1,1]:
                        grid_v[i, j][1] = 0
                    if j == walls[now_level[None],n_wall][0,1]:
                        grid_v[i, j][1] = 0
                    if i == walls[now_level[None],n_wall][0,0]+1:
                        grid_v[i, j][0] = 0
                    if i == walls[now_level[None],n_wall][1,0]-1:
                        grid_v[i, j][0] = 0
                    if j == walls[now_level[None],n_wall][1,1]+1:
                        grid_v[i, j][1] = 0
                    if j == walls[now_level[None],n_wall][0,1]-1:
                        grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v*0.995, new_C*0.995
        x[p] += dt * v[p]  # advection
        # Cut
        if x[p][1] > knife[now_level[None]][0,1] and x[p][1] < knife[now_level[None]][1,1] and abs(x[p][0] - knife[now_level[None]][0,0]) < 0.01:
            if x[p][0] > knife[now_level[None]][0,0]:
                v[p][0] = (0.01 - abs(x[p][0] - knife[now_level[None]][0,0]))*18000
            else:
                v[p][0] = -(0.01 - abs(x[p][0] - knife[now_level[None]][0,0]))*18000

@ti.kernel
def reset():
    for i in range(n_particles):
        x[i] = [
            ((i % size)/size) * 0.16 + pos_start[now_level[None]][0],
            ((i // size)/size) * 0.16 + pos_start[now_level[None]][1]
        ]
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)

gravity[None] = [0, -10]
#level1
pos_start[0] = [10/512, 412/512]
pos_end[0,0] = [120,5]
walls[0,0] = [[0,100],[37,0]]
walls[0,1] = [[95,127],[110,13]]
knife[0] = [[230/512,130/512],[230/512,240/512]]

#level2
pos_start[1] = [10/512, 412/512]
pos_end[1,0] = [122,5]
pos_end[1,1] = [107,5]
pos_end[1,2] = [92,5]
walls[1,0] = [[0,100],[37,0]]
walls[1,1] = [[112,40],[117,0]]
walls[1,2] = [[97,40],[102, 0]]
walls[1,3] = [[82,40],[87, 0]]
knife[1] = [[230/512,130/512],[230/512,240/512]]

#level3
pos_start[2] = [215/512, 428/512]
pos_end[2,0] = [41, 5]
pos_end[2,1] = [86, 5]
walls[2,0] = [[16, 75], [112, 71]]
walls[2,1] = [[0, 45], [32, 0]]
walls[2,2] = [[95, 45], [127, 0]]
walls[2,3] = [[50, 71], [77, 0]]
knife[2] = [[256/512, 300/512],[256/512,382/512]]

reset()
gui = ti.GUI("Acube", res=512, background_color=0xe3e3e3)
while(ifrun):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r':
            reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            ifrun = False
    if gui.event is not None:
        gravity[None] = [0, -10]  # if had any event
    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        gravity[None][0] = -10
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        gravity[None][0] = 10
    if gui.is_pressed(ti.GUI.UP, 'w'):
        gravity[None][1] = 10
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        gravity[None][1] = -10
    for s in range(int(2e-3 // dt)):
        substep()
    # Trigger event
    if now_level[None] == 0:
        if grid_m[pos_end[now_level[None],0][0],pos_end[now_level[None],0][1]] > 1e-5:
            now_level[None] = 1 
            reset()
    if now_level[None] == 1:
        if grid_m[pos_end[now_level[None],0][0],pos_end[now_level[None],0][1]] > 1e-5 \
            and grid_m[pos_end[now_level[None],1][0],pos_end[now_level[None],1][1]] > 1e-5\
            and grid_m[pos_end[now_level[None],2][0],pos_end[now_level[None],2][1]] > 1e-5:
            now_level[None] = 2
            reset()
    if now_level[None] == 2:
        if grid_m[pos_end[now_level[None],0][0],pos_end[now_level[None],0][1]] > 1e-5 \
            and grid_m[pos_end[now_level[None],1][0],pos_end[now_level[None],1][1]] > 1e-5:
            now_level[None] = 3
    for n_wall in range(num_walls):
        topleft = (walls[now_level[None],n_wall][0,0]/128,walls[now_level[None],n_wall][0,1]/128)
        topright = (walls[now_level[None],n_wall][1,0]/128, walls[now_level[None],n_wall][0,1]/128)
        bottomleft = (walls[now_level[None],n_wall][0,0]/128,walls[now_level[None],n_wall][1,1]/128)
        bottomright = (walls[now_level[None],n_wall][1,0]/128,walls[now_level[None],n_wall][1,1]/128)
        gui.triangle(topleft, topright, bottomright,color=0x193c61)
        gui.triangle(topleft, bottomleft, bottomright,color=0x193c61)

    for i in range(num_goals):
        gui.circle((pos_end[now_level[None],i][0]/128, pos_end[now_level[None],i][1]/128) ,color=0x40c930,  radius=8)
    gui.circles(x.to_numpy(), radius=2.5, color=0xED553B)                
    gui.line((knife[now_level[None]][0,0],knife[now_level[None]][0,1]), (knife[now_level[None]][1,0],knife[now_level[None]][1,1]), radius=1, color=0x0a4d8c)
    if now_level[None] == 0:
        gui.text("WASD to move", (0.2,0.9), font_size=18, color=0x4990e6)
        gui.text("R to remake", (0.2,0.85), font_size=18, color=0x4990e6)
        gui.text("knife!", (0.5,0.5), font_size=18, color=0x4990e6)
        gui.text("Come here", (0.85,0.05), font_size=18, color=0x4990e6)
        gui.text("LEVEL 1", (0.5,0.9), font_size=24, color=0x4990e6)
    if now_level[None] == 1:
        gui.text("LEVEL 2", (0.5,0.9), font_size=24, color=0x4990e6)
    if now_level[None] == 2:
        gui.text("LEVEL 3", (0.5,0.9), font_size=24, color=0x4990e6)
    if now_level[None] == 3:
        gui.clear(color=0xe3e3e3)
        gui.text("Finish!", (0.45,0.45), font_size=30, color=0x4990e6)
    gui.show()
