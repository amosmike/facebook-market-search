#%%
import re
import pandas as pd

def get_tabular_data(filepath: str, lineterminator: str = ",") -> pd.DataFrame:
    """
    Function to import data from a csv file and save to a pandas dataframe, dropping all rows with missing data.

    Args:
        filepath (str): string of the path of the file to be imported
        lineterminator (str, optional): string to state the line terminator used in the csv file. Defaults to ",".

    Returns:
        pd.DataFrame: dataframe of the csv contents, with rows with any missing data removed
        
    """
    df = pd.read_csv(filepath, lineterminator=lineterminator).dropna()
    df.rename(columns={'create_time\r':'create_time'}, inplace=True)
    return df


def  clean_price(price: pd.Series) -> pd.Series:
    """
    Function taking a pandas series containing prices to remove all characters that are not digits or '.' and convert values to float.
    
    Args:
        price (pd.Series): pandas series of price data in string format

    Returns:
        pd.Series: pandas series of clean price data in float format
    """
    clean_price = price.replace(to_replace='[^0-9.]', value='', regex=True)
    float_column = pd.to_numeric(clean_price)
    return float_column

def clean_cat(category: pd.Series) -> pd.Series:
    """
    Function taking in a pandas series splits multiple categories and converts it to type 'category'.

    Args:
        category (pd.Series): pandas series of data to be changed to category type

    Returns:
        pd.Series: pandas series of type category
        
    """
    cat_type = category.astype('category')
    clean_cat = cat_type.str.split('/').str[0]
    # clean_cat = category.str.split('/', n=1, expand=True) #.str[0]
    return clean_cat

def cleanse_data(file_path, lineterminator):
    tabular_data = get_tabular_data(file_path, lineterminator)
    tabular_data['price'] = clean_price(tabular_data['price'])
    tabular_data['category'] = clean_cat(tabular_data['category'])
    tabular_data = tabular_data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')) # Removes emojis
    return tabular_data

if __name__ == "__main__":
    file_path = "MetaMarketplaceMLEng/Assets/Products.csv"
    lineterminator = "\n"
    clean_tabular_data = cleanse_data(file_path, lineterminator)
    clean_tabular_data.to_csv("cleaned_products.csv")