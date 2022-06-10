import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

def unique_products_per_customer(data):
    variation = {}
    product = {}
    family = {}
    for cust in data['customer'].unique():
        customer_purchases = data[data['customer'] == cust]
        variation[cust] = customer_purchases['variation'].nunique()
        product[cust] = customer_purchases['product'].nunique()
        family[cust] = customer_purchases['family'].nunique()

    gs = gridspec.GridSpec(1, 3, wspace=0.4)

    pl.figure(figsize=(8, 4))
    pl.tight_layout()
    pl.style.use('seaborn-darkgrid')

    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    pl.hist(variation.values(),bins=list(range(1,420,20)), color='blue', label='unique variations')
    pl.xlabel('variations')
    pl.ylabel('customers')
    pl.ylim(top=1700)
    ax.legend(bbox_to_anchor=(1.05, 1.15))

    ax = pl.subplot(gs[0, 1]) # row 0, col 1
    pl.hist(product.values(),bins=list(range(1,210,10)), color='cornflowerblue', label='unique products')
    pl.xlabel('products')
    pl.ylim(top=1700)
    ax.legend(bbox_to_anchor=(1.05, 1.15))

    ax = pl.subplot(gs[0, 2]) # row 1, span all columns
    pl.hist(family.values(),bins=list(range(1,22,1)), color='slateblue', label='unique family')
    pl.xlabel('families')
    pl.ylim(top=1700)
    ax.legend(bbox_to_anchor=(1, 1.15))

    plt.savefig('unique.png')

if __name__ == '__main__':
    data = pd.read_csv('transaction_history.csv', sep=',', header=0)

    data['family'] = data['prod'].str[:4]
    data['product'] = data['prod'].str[:9]
    data['variation'] = data['prod'].str[:14]
    
    unique_products_per_customer(data)
