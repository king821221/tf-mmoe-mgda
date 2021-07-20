import tensorflow as tf

from util import tf_print


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """

        def case_1():
            gamma = tf.constant(0.999, dtype=v1v1.dtype)
            cost = v1v1
            return gamma, cost

        def case_2():
            gamma = tf.constant(0.001, dtype=v2v2.dtype)
            cost = v2v2
            return gamma, cost

        def case_3():
            gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
            cost = v2v2 + gamma * (v1v2 - v2v2)
            return gamma, cost

        return tf.cond(v1v2 >= v1v1, lambda : case_1(), lambda : tf.cond(v1v2 >= v2v2, lambda : case_2(), lambda : case_3()))

    @staticmethod
    def _min_norm_2d(task_gradients, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dtype = task_gradients[0][0].dtype
        dmin = tf.constant(1e8, dtype=dtype)
        sol = [(-1,-1), tf.constant(0.0, dtype=dtype), tf.constant(0.0, dtype=dtype)]
        for i in range(len(task_gradients)):
            for k in range(len(task_gradients[i])):
                dps[(i, i)] += tf.reduce_sum(
                    tf.multiply(task_gradients[i][k], task_gradients[i][k]))
        for i in range(len(task_gradients)):
            for j in range(i+1,len(task_gradients)):
                assert len(task_gradients[i]) == len(task_gradients[j])
                for k in range(len(task_gradients[i])):
                     dot = tf.reduce_sum(
                        tf.multiply(task_gradients[i][k], task_gradients[j][k]))
                     dps[(i, j)] += dot
                dps[(j, i)] = dps[(i, j)]
                dps[(i, i)] = tf_print(dps[(i, i)],
                                       message = 'dps_i_i-{}-{}'.format(i, i))
                dps[(i, j)] = tf_print(dps[(i, j)],
                                       message = 'dps_i_j-{}-{}'.format(i, j))
                dps[(j, j)] = tf_print(dps[(j, j)],
                                       message = 'dps_j_j-{}-{}'.format(j, j))
                c,d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)])
                c = tf_print(c, message = 'min_norm_2d_{}_{}_c'.format(i, j))
                d = tf_print(d, message = 'min_norm_2d_{}_{}_d'.format(i, j))
                dmin, sol = tf.cond(d < dmin,
                                    lambda : (d, [(i,j), c, d]),
                                    lambda : (dmin, sol))
                dmin = tf_print(dmin, message='dmin')
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = tf.shape(y)[0]
        sorted_y = tf.nn.top_k(y, m)[0]
        sorted_y = tf_print(sorted_y, message = 'projection_sorted_y')
        tmax_f = (tf.reduce_sum(y) - 1.0)/tf.cast(m, dtype=y.dtype)

        tmax_f = tf_print(tmax_f, message = 'tmax_f_default')

        tmax_v = (tf.cumsum(sorted_y) - 1) / tf.cast((tf.range(m)+ 1),
                                                     dtype=sorted_y.dtype)

        sorted_y_v = tf.pad(sorted_y[1:], [[0, 1]])

        tmax_v = tf_print(tmax_v, message = 'tmax_v')

        mask_pad = tf.pad(tf.ones_like(sorted_y[1:], dtype=sorted_y.dtype),
                          [[0,1]], constant_values=0.0)

        tmax_f_mask = tf.cast(tmax_v > sorted_y_v, dtype=tf.int32) * tf.cast(mask_pad, dtype=tf.int32)

        tmax_f_mask = tf_print(tmax_f_mask, message = 'tmax_f_mask')

        print("tmax_f_mask {}".format(tmax_f_mask))

        tmax_f_mask_max = tf.cast(tf.argmax(tmax_f_mask), dtype=tf.int32)

        tmax_f_mask_max = tf_print(tmax_f_mask_max, message = 'tmax_f_mask_max')

        print("tmax_f_mask_max {}".format(tmax_f_mask_max))

        tmax_f = tf.cond(tf.reduce_max(tmax_f_mask) > 0,
                         lambda : tmax_v[tmax_f_mask_max],
                         lambda : tmax_f)

        print("tmax_f {}".format(tmax_f))

        tmax_f = tf_print(tmax_f, message = 'tmax_f')

        return tf.maximum(y - tmax_f, tf.zeros_like(y))

    @staticmethod
    def _next_point(cur_val, grad, n):
        # cur_val: (n,), grad: (n,)
        grad_mean = ( tf.reduce_sum(grad) / n )
        grad_mean = tf_print(grad_mean, message = 'grad_mean', level=0)

        proj_grad = grad - grad_mean

        proj_grad = tf_print(proj_grad, message = 'proj_grad', level=0)

        tm1 = tf.where(proj_grad < 0,
                       -1.0*cur_val/proj_grad,
                       tf.zeros_like(cur_val, dtype=cur_val.dtype))
        tm2 = tf.where(proj_grad > 0,
                       (1.0 - cur_val)/(proj_grad),
                       tf.zeros_like(cur_val,dtype=cur_val.dtype))

        tm1 = tf_print(tm1, message = 'tm1', level=0)
        tm2 = tf_print(tm2, message = 'tm2', level=0)

        tm1_len = tf.reduce_sum(tf.cast(tm1>1e-7, tf.int32))
        tm2_len = tf.reduce_sum(tf.cast(tm2>1e-7, tf.int32))

        tm1_len = tf_print(tm1_len, message = 'tm1_len')
        tm2_len = tf_print(tm2_len, message = 'tm2_len')

        tm1_max = tf.reduce_max(tm1)
        tm2_max = tf.reduce_max(tm2)

        tm1_s = tf.where(tm1>1e-7, tm1, tf.zeros_like(tm1, dtype=tm1.dtype) + tm1_max)
        tm2_s = tf.where(tm2>1e-7, tm2, tf.zeros_like(tm2, dtype=tm2.dtype) + tm2_max)

        tm1_s = tf_print(tm1_s, message = 'tm1_s')
        tm2_s = tf_print(tm2_s, message = 'tm2_s')

        t = tf.cond(tm1_len > 0,
                    lambda : tf.reduce_min(tm1_s),
                    lambda : tf.constant(1.0, dtype=tm1.dtype))
        t = tf.cond(tm2_len > 0,
                    lambda : tf.minimum(tf.reduce_min(tm2_s), t),
                    lambda : t)

        t = tf_print(t, message = 'next_point_t', level=0)

        next_point = proj_grad*t + cur_val # (n,)

        next_point = tf_print(next_point, message = 'next_point')

        next_point = MinNormSolver._projection2simplex(next_point)

        return next_point

    @staticmethod
    def find_min_norm_element(task_gradients):
        """
        task_gradients is composed of
        [[gradient tensors of task 1], [gradient tensors of task 2]]
        each entry is a list of gradients of the specific task loss
        Given a dict of gradient tensors (tensors), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points

        n=len(task_gradients) # number of tasks

        print("number of tasks {}".format(n))

        dtype = task_gradients[0][0].dtype

        grad_mat = {}
        for i in range(n):
            for j in range(n):
                grad_mat[(i, j)] = tf.constant(0.0, dtype=dtype)

        # init_sol: [(i,j),c,d] c is gamma, d is cost
        # dps: {(i,j) -> score}
        init_sol, dps = MinNormSolver._min_norm_2d(task_gradients, grad_mat)

        print("init_sol {} dps {}".format(init_sol, dps))

        assert_op = tf.Assert(
            tf.logical_and(init_sol[0][0] >= 0, init_sol[0][1] >= 0),
            [init_sol[0][0], init_sol[0][1]])

        with tf.control_dependencies([assert_op]):
            init_sol[1] = tf_print(init_sol[1], message = 'init_sol_1')

            sol_vec_i = tf.tile(tf.expand_dims(init_sol[1], 0), [n])
            sol_vec_j = tf.tile(tf.expand_dims(1-init_sol[1], 0), [n])

            i_onehot = tf.one_hot(init_sol[0][0], n)
            j_onehot = tf.one_hot(init_sol[0][1], n)

            i_onehot = tf.cast(i_onehot, dtype=sol_vec_i.dtype)
            j_onehot = tf.cast(j_onehot, dtype=sol_vec_j.dtype)

            print("solv_vec_i {} i_onehot {}".format(sol_vec_i, i_onehot))
            print("solv_vec_j {} j_onehot {}".format(sol_vec_j, j_onehot))

            sol_vec_i = tf_print(sol_vec_i, message='sol_vec_i')
            sol_vec_j = tf_print(sol_vec_j, message='sol_vec_j')

            i_onehot = tf_print(i_onehot, message='i_onehot')
            j_onehot = tf_print(j_onehot, message='j_onehot')

            sol_vec = sol_vec_i * i_onehot + sol_vec_j * j_onehot

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        grad_matrix_vec = []
        for i in range(n):
            for j in range(n):
                grad_matrix_vec.append(dps[(i, j)])
        grad_matrix = tf.stack(grad_matrix_vec)
        assert grad_matrix.shape[0] == n * n, 'INVALID grad matrix shape {}'.format(grad_matrix.shape)
        grad_matrix = tf.reshape(grad_matrix, (n, n))

        tf.summary.histogram('grad_matrix', grad_matrix)

        grad_matrix = tf_print(grad_matrix, message = 'grad_matrix', level=0)

        def cond_func(iter, sol_vec_in, change_in, nd_in):
            iter = tf_print(iter, message = 'loop_iter_cond')
            change_in = tf_print(change_in, message = 'loop_change_in')
            return tf.logical_and(iter < MinNormSolver.MAX_ITER,
                                  change_in >= MinNormSolver.STOP_CRIT)

        def body_func(iter, sol_vec_in, change_in, nd_in):
            iter = tf_print(iter, message = 'loop_iter_body', level=0)
            sol_vec_exp = tf.expand_dims(sol_vec_in, -1) # (n, 1)
            sol_vec_in = tf_print(sol_vec_in, message = 'sol_vec_in', level=0)
            grad_dir = -1.0*tf.matmul(grad_matrix, sol_vec_exp) # (n, 1)
            grad_dir = tf.squeeze(grad_dir, -1) # (n,)
            print("next point sol vec {} grad_dir {} n {}".format(
                sol_vec_in, grad_dir, n
            ))
            grad_dir = tf_print(grad_dir, message = 'grad_dir', level=0)
            new_point = MinNormSolver._next_point(sol_vec_in, grad_dir, n)
            new_point = tf_print(new_point, message = 'new_point', level=0)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec_in[i] * sol_vec_in[j] * dps[(i,j)]
                    v1v2 += sol_vec_in[i] * new_point[j] * dps[(i,j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i,j)]
            v1v1 = tf_print(v1v1, message='v1v1_body')
            v1v2 = tf_print(v1v2, message='v1v2_body')
            v2v2 = tf_print(v2v2, message='v2v2_body')
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            nc = tf_print(nc, message='nc_body')
            nd = tf_print(nd, message='nd_body')
            new_sol_vec = nc*sol_vec_in + (1-nc)*new_point
            change = new_sol_vec - sol_vec_in
            change = tf_print(change, message = 'sol_vec_change')
            change = tf.reduce_sum(tf.abs(change))
            return iter+1, new_sol_vec, change, nd

        iter_count = tf.constant(0)
        init_change = tf.constant(MinNormSolver.STOP_CRIT + 1, dtype=dtype)
        init_nd = tf.constant(0.0, dtype=dtype)

        iter_count, sol_vec, change, nd = tf.while_loop(cond_func,
                                                        body_func,
                                                        [iter_count,
                                                         sol_vec,
                                                         init_change,
                                                         init_nd])

        return sol_vec, nd


    @staticmethod
    def gradient_normalizers(grads, losses, normalization_type='l2'):
        gn = {}
        if normalization_type == 'l2':
            for t in grads.keys():
                gn[t] = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.pow(gr, 2)) for gr in grads[t]]))
        elif normalization_type == 'loss':
            for t in grads:
                gn[t] = losses[t]
        elif normalization_type == 'loss+':
            for t in grads:
                gn[t] = losses[t] * tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.pow(gr, 2)) for gr in grads[t]]))
        elif normalization_type == 'none':
            for t in grads:
                gn[t] = 1.0
        else:
            print('ERROR: Invalid Normalization Type')
        return gn
