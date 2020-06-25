import numpy as np
import matplotlib.pyplot as plt


def minus_log_likelihood(k, lamb):
    return np.log(np.math.factorial(k)) - k * np.log(lamb) + lamb


def main():
    score = np.array([0, 2, 3, 3, 0, 4, 1, 1, 0])
    # score = np.array([0, 3, 1, 0, 0, 1, 0, 2, 0])
    score = score.reshape(3, 3)
    print(score)
    alpha = np.random.rand(3)
    alpha += 1
    beta = np.random.rand(3)
    print(alpha, beta)

    def cal_l():
        all_minus_log_likelihood = 0
        for i in range(3):
            for j in range(3):
                if i != j:
                    lamb = alpha[i] - beta[j] if (alpha[i] - beta[j]) > 0.5 else 0.5
                    all_minus_log_likelihood += minus_log_likelihood(score[i][j], lamb)
        return all_minus_log_likelihood

    print('start training!')
    previous_l, currnt_l = 1, 0
    loss = []
    # while previous_l > currnt_l:
    for _ in range(1000):
        previous_l = cal_l()
        print('log_likelihood:', previous_l)
        print('params before update:', alpha, beta)
        alpha_update = np.array([0, 0, 0])
        beta_update = np.array([0, 0, 0])

        # alpha_update[0] = 3 / (alpha[0] - beta[1]) + 1 / (alpha[0] - beta[2]) - 2
        # alpha_update[1] = 0 / (alpha[1] - beta[0]) + 1 / (alpha[1] - beta[2]) - 2
        # alpha_update[2] = 0 / (alpha[2] - beta[0]) + 2 / (alpha[2] - beta[1]) - 2
        # beta_update[0] = 2 - 0 / (alpha[1] - beta[0]) - 0 / (alpha[2] - beta[0])
        # beta_update[1] = 2 - 3 / (alpha[0] - beta[1]) - 2 / (alpha[2] - beta[1])
        # beta_update[2] = 2 - 1 / (alpha[0] - beta[2]) - 1 / (alpha[1] - beta[2])
        alpha_update[0] = 2 / (alpha[0] - beta[1]) + 3 / (alpha[0] - beta[2]) - 2
        alpha_update[1] = 3 / (alpha[1] - beta[0]) + 4 / (alpha[1] - beta[2]) - 2
        alpha_update[2] = 1 / (alpha[2] - beta[0]) + 1 / (alpha[2] - beta[1]) - 2
        beta_update[0] = 2 - 3 / (alpha[1] - beta[0]) - 1 / (alpha[2] - beta[0])
        beta_update[1] = 2 - 2 / (alpha[0] - beta[1]) - 1 / (alpha[2] - beta[1])
        beta_update[2] = 2 - 3 / (alpha[0] - beta[2]) - 4 / (alpha[1] - beta[2])

        alpha = alpha + 0.001 * alpha_update
        beta = beta + 0.001 * beta_update
        # alpha = alpha - np.mean(alpha) + 1
        # beta = beta - np.mean(beta)
        currnt_l = cal_l()
        print('log_likelihood after update:', previous_l)
        print('params after update:', alpha, beta)
        loss.append(previous_l)

    print('Finish!!!!   log_likelihood:', cal_l())
    plt.figure()
    plt.title("{}  {}  {}".format(np.around(alpha, 2), np.around(beta, 2), np.around(alpha + beta, 2)))
    plt.plot(loss)
    plt.show()

    # we want to minimize the all_minus_log_likelihood



main()
