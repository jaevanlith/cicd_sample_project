import hydra
import pandas as pd
from utils.IO import write_to_csv


@hydra.main(config_path='./config', config_name='main.yaml')
def main(cfg):
    '''
    Main method of repository.

    @param cfg: Hydra configuration object
    '''
    # EXAMPLE: how to use config
    data = {'messages': []}
    for _ in range(cfg.repeat):
        data['messages'].append(cfg.message)

    # EXAMPLE: how to use utils
    df = pd.DataFrame(data)
    write_to_csv(df, cfg.output_filename)


if __name__ == '__main__':
    '''
    Trigger the main function
    ''' 
    main()