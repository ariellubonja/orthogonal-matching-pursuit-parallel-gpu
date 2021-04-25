"""
Use this script because it lets you:

    -Choose which algorithms to run, without editing. Just comment out/in the relevant ones in ALGORITHMS_TO_RUN defn.
    -Choose how many times to run each test (times_to_repeat_tests). Returns Mean,std, mean_err
    -Choose problem (matrix) sizes!
"""


from test_omp import *
# import pandas as pd

# Repeat test a few times to get rid of random variation in system load
# I've optimized them based on alg. speed, but here you can override that
#   E.g. Sklearn is much slower than Cholesky, so you'd want a lower repeat count there
# Change to integer if you want to override. Leave to "default" otherwise
times_to_repeat_tests_override = "default"

# Choose: "SMALL", "MEDIUM", "LARGE", "HUGE"
# Reduce times_to_repeat_tests appropriately
PROBLEM_SIZE = "HUGE"

# Comment these out depending on what you want to run!
ALGORITHMS_TO_RUN = [
    "sklearn",
    "v0_original",
    "v0_new",
    "v0_blas",
    "v0_new_torch",
    "naive_omp"
]

if __name__ == "__main__":
    print("\n\n-----Problem size: ", PROBLEM_SIZE, " -----\n")

    if PROBLEM_SIZE == "SMALL":
        n_components, n_features = 1024, 100
        n_nonzero_coefs = 17
        n_samples = 1000
        times_to_repeat_tests = 10
    elif PROBLEM_SIZE == "MEDIUM":
        n_components, n_features = 3072, 300
        n_nonzero_coefs = 51
        n_samples = 100
        times_to_repeat_tests = 10
    elif PROBLEM_SIZE == "LARGE":
        n_components, n_features = 6144, 600
        n_nonzero_coefs = 102
        n_samples = 40
        times_to_repeat_tests = 3
    elif PROBLEM_SIZE == "HUGE":
        n_components, n_features = 12288, 1200
        n_nonzero_coefs = 204
        n_samples = 10
        times_to_repeat_tests = 2

    if times_to_repeat_tests_override != "default":
        times_to_repeat_tests = times_to_repeat_tests_override

    y, X, w = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0)

    avg_ylen = np.linalg.norm(y, 2, 0)

    # print("Settings used for the test: ")
    print("Number of Components: " + str(n_components))
    print("Number of Features: " + str(n_features))
    print("Number of Nonzero Coefficients: " + str(n_nonzero_coefs))
    print("Number of Samples: " + str(n_samples))
    print("\n")
    print("---Running each test ", times_to_repeat_tests, " times!---")


    if "v0_new_torch" in ALGORITHMS_TO_RUN:
        print('\n\nSingle core. New implementation of algorithm v0 (pytorch)')
        # with torch.autograd.profiler.emit_nvtx():
        results, errors = [], []
        for i in range(times_to_repeat_tests):
            with elapsed_timer() as elapsed:
                xests_v0_new_torch = omp_v0_torch(torch.as_tensor(y.copy()), torch.as_tensor(X.copy()), n_nonzero_coefs)
            # print('Samples per second:', n_samples/elapsed())
            results.append(n_samples/elapsed())
            err_torch = np.linalg.norm(y.T - (X @ xests_v0_new_torch[:, :, None].numpy()).squeeze(-1), 2, 1) / n_samples
            errors.append(err_torch)
        print("---Results for (", times_to_repeat_tests, " repeats)---")
        print("Mean: ", np.mean(results))
        print("Std: ", np.std(results))
        print("Average error: ", np.mean(errors))


    # Optimized Naive implementation (not really Naive though?)
    if "naive_omp" in ALGORITHMS_TO_RUN:
        print('\n\nSingle core. Optimized Naive implementation without Gramian/factorization tricks')
        results, errors = [], []
        for i in range(times_to_repeat_tests):
            with elapsed_timer() as elapsed:
                xests_naive = omp_naive(X.copy(), y.copy(), n_nonzero_coefs)
            # print('Samples per second:', n_samples/elapsed())
            results.append(n_samples/elapsed())
            err_naive = np.linalg.norm(y.T - (X @ xests_naive[:, :, None]).squeeze(-1), 2, 1) / n_samples
            errors.append(err_naive)
        print("---Results for (", times_to_repeat_tests, " repeats)---")
        print("Mean: ", np.mean(results))
        print("Std: ", np.std(results))
        print("Average error: ", np.mean(errors))


    if "v0_original" in ALGORITHMS_TO_RUN:
        print('\n\nSingle core. v0_original algorithm - Cholesky')
        results, errors = [], []
        for i in range(times_to_repeat_tests):
            # Precompute these
            XTX = X.T @ X
            XTy = (X.T @ y.T[:, :, None]).squeeze(-1)
            with elapsed_timer() as elapsed:
                xests_v0 = omp_v0_original(y.copy(), X.copy(), XTX, XTy, n_nonzero_coefs)
            # print('Samples per second:', n_samples/elapsed())
            results.append(n_samples/elapsed())
            err_v0 = np.linalg.norm(y.T - (X @ xests_v0[:, :, None]).squeeze(-1), 2, 1)/ n_samples
            errors.append(err_v0)
        print("---Results for (", times_to_repeat_tests, " repeats)---")
        print("Mean: ", np.mean(results))
        print("Std: ", np.std(results))
        print("Average error: ", np.mean(errors))


    if "v0_new" in ALGORITHMS_TO_RUN:
        print('\n\nSingle core. v0 Cholesky algorithm but much improved with smart updating, memory optimizations')
        results, errors = [], []
        for i in range(times_to_repeat_tests):
            # Precompute these
            XTX = X.T @ X
            XTy = (X.T @ y.T[:, :, None]).squeeze(-1)
            with elapsed_timer() as elapsed:
                xests_v0_new = omp_v0_new(y.copy(), X.copy(), XTX, XTy, n_nonzero_coefs)
            # print('Samples per second:', n_samples/elapsed())
            results.append(n_samples/elapsed())
            err_v0_new = np.linalg.norm(y.T - (X @ xests_v0_new[:, :, None]).squeeze(-1), 2, 1) / n_samples
            errors.append(err_v0_new)
        print("---Results for (", times_to_repeat_tests, " repeats)---")
        print("Mean: ", np.mean(results))
        print("Std: ", np.std(results))
        print("Average error: ", np.mean(errors))


        # with elapsed_timer() as elapsed:
        #
        # print('Samples per second:', n_samples/elapsed())


    if "v0_blas" in ALGORITHMS_TO_RUN:
        print('\n\nSingle core. Improved V0 algorithm - Cholesky + BLAS')
        results, errors = [], []
        for i in range(times_to_repeat_tests):
            # Precompute these
            XTX = X.T @ X
            XTy = (X.T @ y.T[:, :, None]).squeeze(-1)
            with elapsed_timer() as elapsed:
                xests_v0_blas = omp_v0_new_blas(y.copy(), X.copy(), X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
            # print('Samples per second:', n_samples/elapsed())
            results.append(n_samples/elapsed())
            err_v0_blas = np.linalg.norm(y.T - (X @ xests_v0_blas[:, :, None]).squeeze(-1), 2, 1)/ n_samples
            errors.append(err_v0_blas)
        print("---Results for (", times_to_repeat_tests, " repeats)---")
        print("Mean: ", np.mean(results))
        print("Std: ", np.std(results))
        print("Average error: ", np.mean(errors))


    # precompute=True seems slower for single core. Dunno why.
    if "sklearn" in ALGORITHMS_TO_RUN:
        omp_args = dict(n_nonzero_coefs=n_nonzero_coefs, precompute=False, fit_intercept=False)
        print('\n\nSingle core. Sklearn')
        omp = OrthogonalMatchingPursuit(**omp_args)
        results_sklearn, error_sklearn = [], []
        for i in range(times_to_repeat_tests):
            with elapsed_timer() as elapsed:
                omp.fit(X, y)
            # print('Samples per second:', n_samples/elapsed())
            results_sklearn.append(n_samples/elapsed())
            err_sklearn = np.linalg.norm(y.T - (X @ omp.coef_[:, :, None]).squeeze(-1), 2, 1)/ n_samples
            error_sklearn.append(err_sklearn)
            # print("Avg. Error: ", np.average(err_sklearn / avg_ylen))
        print("---Results: Sklearn (", times_to_repeat_tests, " repeats)---")
        print("Mean: ", np.mean(results_sklearn))
        print("Std: ", np.std(results_sklearn))
        print("Average error: ", np.mean(error_sklearn))

    avg_ylen = np.linalg.norm(y, 2, 0)
    # print(np.median(naive_err) / avg_ylen, np.median(scipy_err) / avg_ylen)


    # ----PLOTTING FUNCTIONS------

    # plt.plot(np.sort(err_naive / avg_ylen), label='Naive')
    # plt.plot(np.sort(err_v0 / avg_ylen), '.', label='v0')
    # plt.plot(np.sort(err_v0_new / avg_ylen), '.', label='v0_new')
    # plt.plot(np.sort(err_sklearn / avg_ylen), '--', label='SKLearn')
    # plt.legend()
    # plt.title("Distribution of relative errors.")
    # plt.show()


    # exit(0)
    # # ---------MULTI CORE---------
    # no_workers = 2 # os.cpu_count()
    # # TODO: Gramian can be calculated once locally, and sent to each thread.
    # print('Multi core. With', no_workers, "workers on", os.cpu_count(), "(logical) cores.")
    # inputs = np.array_split(y, no_workers, axis=-1)
    # with multiprocessing.Pool(no_workers, initializer=init_threads, initargs=(solveomp, X, omp_args)) as p:  # num_workers=0
    #     with elapsed_timer() as elapsed:
    #         result = p.map(solveomp, inputs)
    # print('Samples per second:', n_samples / elapsed())

    # dataset = RandomSparseDataset(n_samples, n_components, n_features, n_nonzero_coefs)
    # sampler = torch.utils.data.sampler.BatchSampler(SequentialSampler(dataset), batch_size=2, drop_last=False)

    # y_est = (X @ omp.coef_[:, :, np.newaxis]).squeeze(-1)
    # residuals = np.linalg.norm(y.T - y_est, 2, 0)

    # plt.hist(residuals)
    # plt.show()

