import pandas as pd
import matplotlib.pyplot as plt
import sys


# python dist_cmp.py price_level bucket_id pv_ratio
if __name__ == '__main__':
    key1 = sys.argv[1]
    key2 = sys.argv[2]
    col_name = sys.argv[3]
    price_level_dist = pd.read_csv('~/pyscript/top50_price/price_level.csv', sep='\t')
    data = price_level_dist.groupby([key1, key2, ])[col_name].sum().unstack()
    # plot
    fig = data.plot(kind='bar', figsize=(10, 6), fontsize=15, style=['ro-', 'bs-', 'g^-'])
    font = {'weight': 'normal', 'size': 15}
    fig.legend(prop=font)
    plt.xlabel(key1, font)
    plt.ylabel(col_name, font)
    # plt.title(col_name, font)
    plt.show()
