import numpy as np
from scipy import integrate
from scipy.linalg import expm
import nlopt


def method_nd_md(result_path, par, data_all):
    a1 = par['a1']
    #a2 = 0.5
    #a3 = 1
    L = par['L']

    # theta = 4 * a1 * 1e-8 * L
    #rho = 500
    u = par['u']

    length = len(list(data_all.values())[0])
    

    for i in range(0, length):
        data = [data_all['pop11'][i], data_all['pop22'][i], data_all['pop33'][i], data_all['pop12'][i], data_all['pop13'][i], data_all['pop23'][i]]
        # print(data)
        def func(x,grad):
            if grad.size > 0:
                print("This won't ever happen, since BOBYQA does not use derivatives")
            #m = 0.002
            # unit population size N0
            rho = x[0]

            m12 = x[1]
            m13 = x[2]
            m21 = x[1]
            m23 = x[3]
            m31 = x[2]
            m32 = x[3]

            M12 = x[0] * m12
            M13 = x[0] * m13
            M21 = x[0] * m21
            M23 = x[0] * m23
            M31 = x[0] * m31
            M32 = x[0] * m32

            a2 = x[4]
            a3= x[5]

            A = np.mat([[-(1 / (2 * a1) + 2 * M12 + 2 * M13), 0, 0, 2 * M12, 2 * M13, 0],
                        [0, -(1 / (2 * a2) + 2 * M21 + 2 * M23), 0, 2 * M21, 0, 2 * M23],
                        [0, 0, -(1 / (2 * a3) + 2 * M31 + 2 * M32), 0, 2 * M31, 2 * M32],
                        [M21, M12, 0, -(M12 + M21 + M23 + M13), M23, M13],
                        [M31, 0, M13, M32, -(M31 + M13 + M32 + M12), M12],
                        [0, M32, M23, M31, M21, -(M32 + M23 + M31 + M21)]])
            B = np.mat([[1 / (2 * a1)], [1 / (2 * a2)], [1 / (2 * a3)], [0], [0], [0]])
            E = np.mat([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
            A = A.astype(np.float)
            '''print(A)
            print(A.I)
            print(B)
            print(E)'''

            def func0(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[0]

            def func1(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[1]

            def func2(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[2]

            def func3(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[3]

            def func4(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[4]

            def func5(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[5]

            p_0_r, v0 = integrate.quad(func0, 0, 1)
            p_1_r, v1 = integrate.quad(func1, 0, 1)
            p_2_r, v2 = integrate.quad(func2, 0, 1)
            p_3_r, v3 = integrate.quad(func3, 0, 1)
            p_4_r, v4 = integrate.quad(func4, 0, 1)
            p_5_r, v5 = integrate.quad(func5, 0, 1)
            #print(p_0_r,p_1_r,p_2_r,p_3_r,p_4_r,p_5_r)

            return (p_0_r-data[0])**2*10000+(p_1_r-data[1])**2*10000+(p_2_r-data[2])**2*10000+\
                (p_3_r-data[3])**2*10000+(p_4_r-data[4])**2*10000+(p_5_r-data[5])**2*10000


        # def f(x,grad):
        #     if grad.size > 0:
        #         print("This won't ever happen, since BOBYQA does not use derivatives")
        #     return (x-2.0)**2

        opt = nlopt.opt(nlopt.LN_BOBYQA,6)
        opt.set_min_objective(func)
        #stopval = 0.01
        #opt.set_stopval(stopval)
        tol = 0.00001
        opt.set_xtol_rel(tol)
        #opt.get_xtol_rel()
        opt.set_lower_bounds(par["lower_bounds"])
        opt.set_upper_bounds(par["upper_bounds"])
        x = par["init_value"]
        xopt = opt.optimize(x)
        minf = opt.last_optimum_value()
        with open(result_path, 'a') as f:
            print("Result:", i, file=f)
            print("population1 size:", xopt[0],file=f)
            print("population2 size", xopt[0]*xopt[4],file=f)
            print("population3 size",xopt[0]*xopt[5],file=f)
            print("migration rate m12-m21", xopt[1], file=f)
            print("migration rate m13-m31", xopt[2], file=f)
            print("migration rate m23-m32", xopt[3], file=f)
            print("minimum value = ", minf, file=f)
            print('\n', file=f)
            # print("N2",'250',file=f)
            # print("N3",'500',file=f)
            # print("optimum at ",xopt,file=f)
        # print("optimum at ",xopt)
        # print("minimum value = ", minf)
        # print("result code = ", opt.last_optimize_result())
            


def method_nd_ms(result_path, par, data_all):

    a1 = par['a1']
    #N2 = 0.5
    #N3 = 1
    L = par['L']
    theta = 4 * a1 * 1e-8 * L
    #rho = 500
    u = par['u']

    # data nd_ms N1=500 N2=250 N3=500 n=500 m=0.002-0.03

    length = len(list(data_all.values())[0])
    

    for i in range(0, length):
        data = [data_all['pop11'][i], data_all['pop22'][i], data_all['pop33'][i], data_all['pop12'][i], data_all['pop13'][i], data_all['pop23'][i]]
        print(data)
        def func(x,grad):
            if grad.size > 0:
                print("This won't ever happen, since BOBYQA does not use derivatives")
            #m = 0.002
            rho = x[0]

            m12 = x[1]
            m13 = x[1]
            m21 = x[1]
            m23 = x[1]
            m31 = x[1]
            m32 = x[1]

            M12 = x[0] * m12
            M13 = x[0] * m13
            M21 = x[0] * m21
            M23 = x[0] * m23
            M31 = x[0] * m31
            M32 = x[0] * m32

            a2 = x[2]
            a3 = x[3]

            A = np.mat([[-(1 / (2 * a1) + 2 * M12 + 2 * M13), 0, 0, 2 * M12, 2 * M13, 0],
                        [0, -(1 / (2 * a2) + 2 * M21 + 2 * M23), 0, 2 * M21, 0, 2 * M23],
                        [0, 0, -(1 / (2 * a3) + 2 * M31 + 2 * M32), 0, 2 * M31, 2 * M32],
                        [M21, M12, 0, -(M12 + M21 + M23 + M13), M23, M13],
                        [M31, 0, M13, M32, -(M31 + M13 + M32 + M12), M12],
                        [0, M32, M23, M31, M21, -(M32 + M23 + M31 + M21)]])
            B = np.mat([[1 / (2 * a1)], [1 / (2 * a2)], [1 / (2 * a3)], [0], [0], [0]])
            E = np.mat([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
            A = A.astype(np.float)
            '''print(A)
            print(A.I)
            print(B)
            print(E)'''

            def func0(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[0]

            def func1(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[1]

            def func2(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[2]

            def func3(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[3]

            def func4(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[4]

            def func5(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[5]

            p_0_r, v0 = integrate.quad(func0, 0, 1)
            p_1_r, v1 = integrate.quad(func1, 0, 1)
            p_2_r, v2 = integrate.quad(func2, 0, 1)
            p_3_r, v3 = integrate.quad(func3, 0, 1)
            p_4_r, v4 = integrate.quad(func4, 0, 1)
            p_5_r, v5 = integrate.quad(func5, 0, 1)
            #print(p_0_r,p_1_r,p_2_r,p_3_r,p_4_r,p_5_r)

            return (p_0_r-data[0])**2*10000+(p_1_r-data[1])**2*10000+(p_2_r-data[2])**2*10000+\
                (p_3_r-data[3])**2*10000+(p_4_r-data[4])**2*10000+(p_5_r-data[5])**2*10000

        opt = nlopt.opt(nlopt.LN_BOBYQA,4)
        opt.set_min_objective(func)
        #stopval = 0.01
        #opt.set_stopval(stopval)
        tol = 0.00001
        opt.set_xtol_rel(tol)
        #opt.get_xtol_rel()
        opt.set_lower_bounds(par["lower_bounds"])
        opt.set_upper_bounds(par["upper_bounds"])
        x = par["init_value"]
        xopt = opt.optimize(x)
        minf = opt.last_optimum_value()
        with open(result_path, 'a') as f:
            print("Result:", i, file=f)
            print("population1 size:", xopt[0],file=f)
            print("population2 size", xopt[0]*xopt[2],file=f)
            print("population3 size",xopt[0]*xopt[3],file=f)
            print("migration rate m", xopt[1], file=f)
            print("minimum value = ", minf, file=f)
            print('\n', file=f)
        #print("minimum value = ", minf)
        #print("result code = ", opt.last_optimize_result())
        # print("optimum at ",xopt)
        # print("minimum value = ", minf)
        # print("result code = ", opt.last_optimize_result())
            

def method_ns_md(result_path, par, data_all):

    a1 = par['a1']
    a2 = par['a2']
    a3 = par['a3']
    L = par['L']
    theta = 4 * a1 * 1e-8 * L
    #rho = 500
    u = par['u']

    # data ns_md n=500 m=0.002-0.03
    length = len(list(data_all.values())[0])
    

    for i in range(0, length):
        data = [data_all['pop11'][i], data_all['pop22'][i], data_all['pop33'][i], data_all['pop12'][i], data_all['pop13'][i], data_all['pop23'][i]]
        # print(data)
        def func(x,grad):
            if grad.size > 0:
                print("This won't ever happen, since BOBYQA does not use derivatives")
            #m = 0.002
            rho = x[0]

            m12 = x[1]
            m13 = x[2]
            m21 = x[1]
            m23 = x[3]
            m31 = x[2]
            m32 = x[3]

            M12 = x[0] * m12
            M13 = x[0] * m13
            M21 = x[0] * m21
            M23 = x[0] * m23
            M31 = x[0] * m31
            M32 = x[0] * m32

            A = np.mat([[-(1 / (2 * a1) + 2 * M12 + 2 * M13), 0, 0, 2 * M12, 2 * M13, 0],
                        [0, -(1 / (2 * a2) + 2 * M21 + 2 * M23), 0, 2 * M21, 0, 2 * M23],
                        [0, 0, -(1 / (2 * a3) + 2 * M31 + 2 * M32), 0, 2 * M31, 2 * M32],
                        [M21, M12, 0, -(M12 + M21 + M23 + M13), M23, M13],
                        [M31, 0, M13, M32, -(M31 + M13 + M32 + M12), M12],
                        [0, M32, M23, M31, M21, -(M32 + M23 + M31 + M21)]])
            B = np.mat([[1 / (2 * a1)], [1 / (2 * a2)], [1 / (2 * a3)], [0], [0], [0]])
            E = np.mat([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
            A = A.astype(np.float)
            '''print(A)
            print(A.I)
            print(B)
            print(E)'''

            def func0(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[0]

            def func1(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[1]

            def func2(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[2]

            def func3(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[3]

            def func4(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[4]

            def func5(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[5]

            p_0_r, v0 = integrate.quad(func0, 0, 1)
            p_1_r, v1 = integrate.quad(func1, 0, 1)
            p_2_r, v2 = integrate.quad(func2, 0, 1)
            p_3_r, v3 = integrate.quad(func3, 0, 1)
            p_4_r, v4 = integrate.quad(func4, 0, 1)
            p_5_r, v5 = integrate.quad(func5, 0, 1)
            #print(p_0_r,p_1_r,p_2_r,p_3_r,p_4_r,p_5_r)

            return (p_0_r-data[0])**2*10000+(p_1_r-data[1])**2*10000+(p_2_r-data[2])**2*10000+\
                (p_3_r-data[3])**2*10000+(p_4_r-data[4])**2*10000+(p_5_r-data[5])**2*10000

        opt = nlopt.opt(nlopt.LN_BOBYQA,4)
        opt.set_min_objective(func)
        #stopval = 0.01
        #opt.set_stopval(stopval)
        tol = 0.00001
        opt.set_xtol_rel(tol)
        #opt.get_xtol_rel()
        opt.set_lower_bounds(par["lower_bounds"])
        opt.set_upper_bounds(par["upper_bounds"])
        x = par["init_value"]
        xopt = opt.optimize(x)
        minf = opt.last_optimum_value()
        with open(result_path, 'a') as f:
            print("Result:", i, file=f)
            print("population size:", xopt[0],file=f)
            print("migration rate m12-m21", xopt[1], file=f)
            print("migration rate m13-m31", xopt[2], file=f)
            print("migration rate m23-m32", xopt[3], file=f)
            print("minimum value = ", minf, file=f)
            print('\n', file=f)
        # print("optimum at ",xopt)
        # print("minimum value = ", minf)
        # print("result code = ", opt.last_optimize_result())
        

def method_ns_ms(result_path, par, data_all):
    a1 = par['a1']
    a2 = par['a2']
    a3 = par['a3']
    L = par['L']
    theta = 4 * a1 * 1e-8 * L
    #rho = 500
    u = par['u']


    # data ns_ms n=500 m=0.002-0.03
    length = len(list(data_all.values())[0])
    

    for i in range(0, length):
        data = [data_all['pop11'][i], data_all['pop22'][i], data_all['pop33'][i], data_all['pop12'][i], data_all['pop13'][i], data_all['pop23'][i]]
        # print(data)


        def func(x,grad):
            if grad.size > 0:
                print("This won't ever happen, since BOBYQA does not use derivatives")
            #m = 0.002
            rho = x[0]

            m12 = x[1]
            m13 = x[1]
            m21 = x[1]
            m23 = x[1]
            m31 = x[1]
            m32 = x[1]

            M12 = x[0] * m12
            M13 = x[0] * m13
            M21 = x[0] * m21
            M23 = x[0] * m23
            M31 = x[0] * m31
            M32 = x[0] * m32

            A = np.mat([[-(1 / (2 * a1) + 2 * M12 + 2 * M13), 0, 0, 2 * M12, 2 * M13, 0],
                        [0, -(1 / (2 * a2) + 2 * M21 + 2 * M23), 0, 2 * M21, 0, 2 * M23],
                        [0, 0, -(1 / (2 * a3) + 2 * M31 + 2 * M32), 0, 2 * M31, 2 * M32],
                        [M21, M12, 0, -(M12 + M21 + M23 + M13), M23, M13],
                        [M31, 0, M13, M32, -(M31 + M13 + M32 + M12), M12],
                        [0, M32, M23, M31, M21, -(M32 + M23 + M31 + M21)]])
            B = np.mat([[1 / (2 * a1)], [1 / (2 * a2)], [1 / (2 * a3)], [0], [0], [0]])
            E = np.mat([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
            A = A.astype(np.float)
            '''print(A)
            print(A.I)
            print(B)
            print(E)'''

            def func0(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[0]

            def func1(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[1]

            def func2(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[2]

            def func3(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[3]

            def func4(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[4]

            def func5(t):
                res1 = A.I * (expm(t * A) - E) * B
                res2 = A * res1 + B
                return (2 * rho * u * t * np.exp(-2 * rho * u * t) + np.exp(-2 * rho * u * t)) * res2[5]

            p_0_r, v0 = integrate.quad(func0, 0, 1)
            p_1_r, v1 = integrate.quad(func1, 0, 1)
            p_2_r, v2 = integrate.quad(func2, 0, 1)
            p_3_r, v3 = integrate.quad(func3, 0, 1)
            p_4_r, v4 = integrate.quad(func4, 0, 1)
            p_5_r, v5 = integrate.quad(func5, 0, 1)
            #print(p_0_r,p_1_r,p_2_r,p_3_r,p_4_r,p_5_r)

            return (p_0_r-data[0])**2*10000+(p_1_r-data[1])**2*10000+(p_2_r-data[2])**2*10000+\
                (p_3_r-data[3])**2*10000+(p_4_r-data[4])**2*10000+(p_5_r-data[5])**2*10000

        opt = nlopt.opt(nlopt.LN_BOBYQA,2)
        opt.set_min_objective(func)
        #stopval = 0.01
        #opt.set_stopval(stopval)
        tol = 0.00001
        opt.set_xtol_rel(tol)
        #opt.get_xtol_rel()
        opt.set_lower_bounds(par["lower_bounds"])
        opt.set_upper_bounds(par["upper_bounds"])
        x = par["init_value"]
        xopt = opt.optimize(x)
        minf = opt.last_optimum_value()
        with open(result_path, 'a') as f:
            print("Result:", i, file=f)
            print("population size:", xopt[0],file=f)
            print("migration rate m12-m21", xopt[1], file=f)
            print("minimum value = ", minf, file=f)
            print('\n', file=f)
        
        #print("minimum value = ", minf)
        #print("result code = ", opt.last_optimize_result())
            

            