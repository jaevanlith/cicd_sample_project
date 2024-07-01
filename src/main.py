import hydra
from train import train
from predict import predict
from utils import IO


@hydra.main(config_path='./conf', config_name='main.yaml', version_base='1.1')
def main(cfg):
    '''
    Main method of repository.

    @param cfg: Hydra configuration object
    '''
    print('Running main...')

    model = None

    if cfg.train:
        print('Training model...')
        train_score, test_score, model = train(cfg)
        print(f'Train score: {train_score}')
        print(f'Test score: {test_score}')
    
    if cfg.predict:
        print('Predicting...')
        if model is None:
            model = IO.load_model(cfg.model_dir + '/' + cfg.model + '.pkl')
        predict(cfg, model)


if __name__ == '__main__':
    '''
    Trigger the main function
    ''' 
    main()