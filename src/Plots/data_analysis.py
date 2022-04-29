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

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    pl.figure()
    pl.tight_layout()

    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    pl.hist(variation.values(),bins=list(range(0,420,20)), color='blue', label='unique variations')
    pl.xlabel('variations')
    pl.ylabel('customers')
    ax.legend(bbox_to_anchor=(2.2, -0.45))

    ax = pl.subplot(gs[0, 1]) # row 0, col 1
    pl.hist(product.values(),bins=list(range(0,210,10)), color='cornflowerblue', label='unique products')
    pl.xlabel('products')
    pl.ylabel('customers')
    ax.legend(bbox_to_anchor=(0.76, -0.72))

    ax = pl.subplot(gs[1, 0]) # row 1, span all columns
    pl.hist(family.values(),bins=range(0,20,1), color='slateblue', label='unique family')
    pl.xlabel('families')
    pl.ylabel('customers')
    ax.legend(bbox_to_anchor=(2.08, 0.3))

    pl.suptitle('Product analysis per customer')
    plt.savefig('unique.png')

if __name__ == '__main__':
    data = pd.read_csv('transaction_history.csv', sep=',', header=0)

    data['family'] = data['prod'].str[:4]
    data['product'] = data['prod'].str[:9]
    data['variation'] = data['prod'].str[:14]
    
    unique_products_per_customer(data)
