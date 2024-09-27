# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


def select_needed_columns(df:pd.DataFrame, output_filepath)->pd.DataFrame:
    
    df = df.assign( genre_len = lambda x:len(x['Genres']))
    print(df)
    df = df[df['genre_len']>0]
    df = df[['Book','Genres','Description']]
    df.to_csv(output_filepath, index=False)
     

# add function to load data, clean data, and save data

def load_data(input_filepath:str)->pd.DataFrame:
    """ Load data from input_filepath
    """
    data = pd.read_csv(input_filepath)
    print(data.head())
    return data
    
    # df = pd.read_csv('data/raw/goodreads_data.csv').drop(['Unnamed: 0'],axis=1)




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    loaded_df = load_data(input_filepath)
    select_needed_columns(loaded_df, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
