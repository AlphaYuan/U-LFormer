import os
import csv
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def load_from_trajectory_results(root, exp_range=None, fov_range=None):
    results = []
    return_num = 10
    # exp_range = range(12)
    exp_range = list(range(1, 9 + 1)) if exp_range is None else exp_range
    for exp in exp_range:
        # os.makedirs(os.path.join(args.output_dir, 'exp_{}'.format(exp)), exist_ok=True)
        fov_range = range(30) if fov_range is None else fov_range
        fov_res = {'alpha': [], 'K': [], 'weight': []}
        for fov in fov_range:
            # print("Exp {}, FOV {}".format(exp, fov))
            file_path = os.path.join(root, 'exp_{}'.format(exp), 'fov_{}.txt'.format(fov))
            print('{} exists: '.format(file_path), os.path.exists(file_path))
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_split = line.rstrip().split(',')[1:]
                    previous_timestep = 0
                    for i in range(0, len(line_split), 4):
                        fov_res['K'].append(float(line_split[i]))
                        fov_res['alpha'].append(float(line_split[i + 1]))
                        time_spent = int(line_split[i + 3]) - previous_timestep
                        fov_res['weight'].append(time_spent)
                        previous_timestep = int(line_split[i + 3])
                    # print(line_split, fov_res['weight'])
                    # return_num -= 1
                    # if return_num == 0:
                    #     return
        results.append(fov_res)
    return results


def bgmm_fit(a, K_max=30, feature_dim=1, progressive_explore=False, time_weight=None):
    # Set the maximum component for BayesianGMM
    max_components = K_max
    bgmm = BayesianGaussianMixture(n_components=max_components, random_state=42)
    bgmm.fit(a.reshape(-1, 1))

    # get weights
    means = bgmm.means_.flatten()
    covariances = bgmm.covariances_.flatten()
    weights = bgmm.weights_

    # filter out small components
    threshold = 1e-3
    valid_components = weights > threshold
    filtered_means = means[valid_components]
    filtered_covariances = covariances[valid_components]
    filtered_weights = weights[valid_components]

    # print("根据BIC选择的最佳模型的估计均值: ", filtered_means)
    # print("根据BIC选择的最佳模型的估计方差: ", filtered_covariances**2)
    # print("根据BIC选择的最佳模型的估计权重: ", filtered_weights)

    # For every data point, calculate its posterior probabilities
    posterior_probs = bgmm.predict_proba(a.reshape(-1, 1))

    estimated_means = np.dot(posterior_probs, means)
    estimated_variances = np.dot(posterior_probs, covariances)

    return estimated_means, estimated_variances

def gmm_fit(a, K_max=30, feature_dim=1, progressive_explore=False, K_given=None, time_weight=None):
    max_components = K_max  # Set the maximum component for GMM

    bic_scores = []
    aic_scores = []
    models = []

    if time_weight is not None:
        print(time_weight.shape, a.shape)
        print("加权 Gaussian Mixture Model")
        a = np.repeat(a, time_weight, axis=0)

    # Try different GMM component
    if K_given is not None:
        for n_components in [K_given]:
            gmm = GaussianMixture(n_components=n_components, max_iter=100, random_state=42)
            gmm.fit(a.reshape(-1, 1))
            bic_scores.append(gmm.bic(a.reshape(-1, 1)))
            models.append(gmm)
        best_bic_index = np.argmin(bic_scores)
        best_gmm_bic = models[best_bic_index]
    elif not progressive_explore:
        for n_components in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n_components, max_iter=100, random_state=42)
            gmm.fit(a.reshape(-1, 1))
            bic_scores.append(gmm.bic(a.reshape(-1, 1)))
            aic_scores.append(gmm.aic(a.reshape(-1, 1)))
            models.append(gmm)

        # get the model with smallest BIC
        best_bic_index = np.argmin(bic_scores)
        best_gmm_bic = models[best_bic_index]
    else:
        n_components = 1
        while True:
            gmm = GaussianMixture(n_components=n_components, max_iter=100, random_state=42)
            gmm.fit(a.reshape(-1, 1))
            bic_scores.append(gmm.bic(a.reshape(-1, 1)))
            aic_scores.append(gmm.aic(a.reshape(-1, 1)))
            models.append(gmm)

            # get the model with smallest BIC
            best_bic_index = np.argmin(bic_scores)
            best_gmm_bic = models[best_bic_index]
            
            if best_bic_index + 1 == max_components:
                # print(f"增加component数量, best: {best_bic_index}, cur: {n_components}, max: {max_components}")
                max_components += 1
            elif n_components == max_components:
                break
            n_components += 1

    # print(f"根据BIC选择的最佳模型的高斯分布数量: {best_bic_index + 1}")

    # Choose optimal GMM weights according to BIC
    mu_estimates = best_gmm_bic.means_.flatten()
    sigma_estimates = np.sqrt(best_gmm_bic.covariances_).flatten()
    weights = best_gmm_bic.weights_

    print("根据BIC选择的最佳模型的估计均值: ", mu_estimates)
    print("根据BIC选择的最佳模型的估计方差: ", sigma_estimates**2)
    print("根据BIC选择的最佳模型的估计权重: ", weights)

    # For every data point, calculate its posterior probabilities
    responsibilities = best_gmm_bic.predict_proba(a.reshape(-1, 1))

    # For every data point, calculate its gaussian distribution
    assigned_distributions = np.argmax(responsibilities, axis=1)

    # Assign the mean and std to each data point
    assigned_mu = mu_estimates[assigned_distributions]
    assigned_sigma = sigma_estimates[assigned_distributions]

    return assigned_mu, assigned_sigma


def get_weight(assigned_mu, assigned_sigma, fov_res, time_weighted=False):
    # weight = np.zeros((np.unique(assigned_mu).shape[0], 1))
    weight_map = {}
    time_sum = 0
    for i in range(assigned_mu.shape[0]):
        key = '{}_{}_{}_{}'.format(assigned_mu[i][0], assigned_mu[i][1], assigned_sigma[i][0], assigned_sigma[i][1],)
        if key not in weight_map:
            weight_map[key] = 0
        weight_map[key] += fov_res['weight'][i] if not time_weighted else 1
        time_sum += fov_res['weight'][i] if not time_weighted else 1
    print(weight_map)
    results = [] 
    for key, value in weight_map.items():
        keys = key.split('_')
        # print(keys)
        results.append([float(keys[0]), float(keys[2]), float(keys[1]), float(keys[3]), value / time_sum])
    return results


def analyse_ensemble(root, results, K_max=3, output_dir=None):
    print(len(results))
    exp_range = list(range(1, 9 + 1)) # start from one
    # K_map = {0: (2, 2), 1: (2, 2), 2: (2, 2), 3: (4, 4), 4: (2, 2), 5: (3, 3), 6: (2, 2), 7: (2, 2), 8: (2, 2), 9: (2, 2), 10: (3, 3), 11: (2, 2)}
    # exp_range = [11]
    for exp in exp_range:
        K_max_a, K_max_k = 5, 4
        # K_max_a, K_max_k = K_map[exp]
        print('Processing: ', exp)
        fov_res = results[exp]
        alpha = np.array(fov_res['alpha']).reshape(-1, )
        K = np.array(fov_res['K']).reshape(-1, )
        weights = None #np.array(fov_res['weight']).reshape(-1, )
        assigned_mu_alpha, assigned_sigma_alpha = gmm_fit(alpha, K_max=K_max_a, time_weight=weights)
        assigned_mu_K, assigned_sigma_K = gmm_fit(K, K_max=K_max_k, time_weight=weights)
        # joint = np.concatenate((alpha, K), axis=-1)
        # assigned_mu, assigned_sigma = gmm_fit(joint, K_max=5)
        cat_mu = np.concatenate((assigned_mu_alpha.reshape(-1,1), assigned_mu_K.reshape(-1,1)), axis=-1)
        cat_sigma = np.concatenate((assigned_sigma_alpha.reshape(-1,1), assigned_sigma_K.reshape(-1,1)), axis=-1)
        ensemble_results = get_weight(cat_mu, cat_sigma, fov_res, time_weighted=False if weights is None else True)
        # 从 1 开始
        # ensemble_results = [ensemble_results[0]] + ensemble_results
        print('num: ', len(ensemble_results))
        if output_dir is None:
            output_dir = root
        file_path = os.path.join(output_dir, 'exp_{}'.format(exp), 'ensemble_labels.txt')
        with open(file_path, 'w') as f:
            f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
            writer = csv.writer(f, delimiter=';')
            # writer.writerows(ensemble_results)
            writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose]
        
        # os.makedirs(os.path.join(output_dir, 'ensemble', 'exp_{}'.format(exp)), exist_ok=True)
        # file_path = os.path.join(output_dir, 'ensemble', 'exp_{}'.format(exp), 'ensemble_labels.txt')
        # with open(file_path, 'w') as f:
        #     f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
        #     writer = csv.writer(f, delimiter=';')
        #     # writer.writerows(ensemble_results)
        #     writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose


if __name__ == '__main__':
    # load_from_trajectory_results('/data4/jiangy/AnDiChallenge/dataset/public_data_validation_v1/track_2/')
    # root = '/data4/jiangy/AnDiChallenge/andi_solver/outputs_0602/test_andi_validation/track_2_new_GatedLSTM_999'
    # root = '/data4/jiangy/AnDiChallenge/andi_solver/outputs_0605/test_andi_validation/track_1_all_GatedLSTM_MAEa_MSEk_loglabel_209_segalpha'
    # root = '../../results/track_1_all'
    # root = '../../challenge_reults/9VA/track_2'
    # root = '/data1/jiangy/andi_tcu/challenge_results/9VA/track_2'
    # root = '/data1/jiangy/andi_tcu/challenge_results/daR_new/track_2'
    # root = '/data1/jiangy/andi_tcu/challenge_results/0706/daR/track_1_all'
    # output_dir = '/data1/jiangy/andi_tcu/challenge_results/0707/daR/track_1_all'
    root = '/data1/jiangy/andi_tcu/challenge_results/0713/track_1_all'
    output_dir = '/data1/jiangy/andi_tcu/challenge_results/0713/track_1_all'
    results = load_from_trajectory_results(root)
    analyse_ensemble(root, results, K_max=5, output_dir=output_dir)
