import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
import time


def minus_log_likelihood(k, lamb):
    return np.log(np.math.factorial(k)) - k * np.log(lamb) + lamb


def main():
    # load score data
    df = pd.read_csv('output.csv').drop(columns=['Unnamed: 0'])

    d = LabelEncoder()
    df['home_number'] = d.fit_transform(df['home'])
    df['away_number'] = d.transform(df['away'])

    res = pd.DataFrame({'team': list(set(df['home']))})
    res['number'] = d.transform(res['team'])

    alpha = np.random.rand(20)
    alpha += 1
    beta = np.random.rand(20)
    print(alpha, beta)

    def cal_l():
        all_minus_log_likelihood = 0
        for index, row in df.iterrows():
            # calculate home score likelihood
            if alpha[row['home_number']] - beta[row['away_number']] > 0.5:
                lamb = alpha[row['home_number']] - beta[row['away_number']]
            else:
                lamb = 0.5
            all_minus_log_likelihood += minus_log_likelihood(row['home_score'], lamb)

            # calculate away score likelihood
            if alpha[row['away_number']] - beta[row['home_number']] > 0.5:
                lamb = alpha[row['away_number']] - beta[row['home_number']]
            else:
                lamb = 0.5
            all_minus_log_likelihood += minus_log_likelihood(row['away_score'], lamb)
        return all_minus_log_likelihood

    print(cal_l())
    print('Start training!')
    loss = []

    def gradient_alpha(i):  # update from goals
        gradient = 0
        for index, row in df[df['home_number'] == i].iterrows():
            gradient += row['home_score']/(alpha[i] - beta[row['away_number']]) - 1
        for index, row in df[df['away_number'] == i].iterrows():
            gradient += row['away_score'] / (alpha[i] - beta[row['home_number']]) - 1
        return gradient

    def gradient_beta(i):  # update from goals conceded
        gradient = 0
        for index, row in df[df['home_number'] == i].iterrows():
            gradient += 1 - row['away_score']/(alpha[row['away_number']] - beta[i])
        for index, row in df[df['away_number'] == i].iterrows():
            gradient += 1 - row['home_score'] / (alpha[row['home_number']] - beta[i])
        return gradient

    for _ in range(100):
        previous_l = cal_l()
        # print('log_likelihood:', previous_l)
        # print('params before update:', alpha, beta)
        alpha_update, beta_update = np.zeros(20), np.zeros(20)
        for i in range(20):
            alpha_update[i] = gradient_alpha(i)
            beta_update[i] = gradient_beta(i)
        alpha, beta = alpha + 0.001 * alpha_update, beta + 0.001 * beta_update
        currnt_l = cal_l()
        print('log_likelihood after update:', currnt_l)
        print()
        loss.append(previous_l)

    print('Finish!!!!   log_likelihood:', cal_l())
    plt.figure()
    plt.title('Minus log likelihood')
    plt.plot(loss)
    plt.show()
    res['attack'] = res['number'].apply(lambda x: round(alpha[x], 2))
    res['defense'] = res['number'].apply(lambda x: round(beta[x], 2))
    res['sum'] = res['number'].apply(lambda x: round(alpha[x]+beta[x], 2))
    res = res.sort_values(by=['sum'], ascending=False).drop(columns=['number']).reset_index(drop=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res)


main()
