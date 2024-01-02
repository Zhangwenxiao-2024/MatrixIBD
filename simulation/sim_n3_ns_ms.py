import sys, os


# def worker(cmd, i):
#    print('job-{}: {}'.format(i,cmd))
#    os.system(cmd)

def auto_gen(N1, N2, N3, L=100000000):
    jobs = []
    for i in range(1, 16):
        theta = 4 * N1 * 1e-8 * L
        rho = theta
        m = 0.002 * i
        m12 = m
        m13 = m
        m21 = m
        m23 = m
        m31 = m
        m32 = m
        M12 = 4 * N1 * m12
        M13 = 4 * N1 * m13
        M21 = 4 * N1 * m21
        M23 = 4 * N1 * m23
        M31 = 4 * N1 * m31
        M32 = 4 * N1 * m32
        rate2 = N2 / N1
        rate3 = N3 / N1
        if N1 == N2 == N3:
            cmd = './ms 90 20 -t {} -r {} {} -I 3 30 30 30 -m 1 2 {} -m 2 1 {} -m 1 3 {} -m 3 1 {} -m 2 3 {} -m 3 2 {} -p 8 > Ns_ms_n3_{}.txt '.format(
                theta, rho, L, M12, M21, M13, M31, M23, M32, i)
        else:
            cmd = './ms 90 20 -t {} -r {} {} -I 3 30 30 30 -m 1 2 {} -m 2 1 {} -m 1 3 {} -m 3 1 {} -m 2 3 {} -m 3 2 {} -n 2 {} -n 3 {} -p 8 > Nd_ms_n3_{}.txt '.format(
                theta, rho, L, M12, M21, M13, M31, M23, M32, rate2, rate3, i)
        print(cmd)
        os.system(cmd)
    # print('return status: {}'.format(ret))
    # p = multiprocessing.Process(name='job-'+repr(i),target=worker, args=(cmd,i))
    # jobs.append(p)
    # p.start()


if __name__ == '__main__':
    if len(sys.argv) > 4:
        auto_gen(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) > 3:
        auto_gen(int(sys.argv[1]), int(sys.argv[3]))
    else:
        print('Usage: python {} N1 N2 N3 L'.format(sys.argv[0]))


