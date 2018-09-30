from gym.envs.registration import register
# /home/{USER}/.local/lib/python3.5/site-packages/gym/envs/__init__.py
# /home/lucas/.local/lib/python3.5/site-packages/gym/envs/__init__.py
register(
    id='EuroDolTrain-v0',
    entry_point='trading_env.eurodol_train:EuroDol',
    kwargs={"filename": "EURUSD_5.csv"}
)

register(
    id='EuroDolEval-v0',
    entry_point='trading_env.eurodol_eval:EuroDol',
    kwargs={"filename": "EURUSD_5.csv"}
)
