import torch
class transformerconfig:
    def __init__(self) -> None:
        self.POS_EMBEDDING = "learnt2_c"

        self.use_other_obs_encoder = False

        self.condition_decoder=False

        self.controller_type = "daab"

        self.daab_observation_encoder_mode = "full_obs"

        self.daab_position_embedding_mode = "axis_concat"

        self.daab_bias_feature_type = "5d"

        self.daab_bias_scale = 1.0

        self.daab_bias_clip = 10.0

        self.daab_bias_layers = 2

        self.daab_bias_hidden_dim = 32

        self.daab_center_scale = 1.0

        self.daab_velocity_scale = 1.0

        self.use_separate_pos_embedding=True

        self.attention_embedding_size=64

        self.attention_heads=1

        self.daab_attention_heads=1

        self.attention_hidden_size=128

        self.attention_layers=1

        #loss=lambda_a * attention_kl + lambda_h * feature_mse
        self.attention_distill_loss = "attention_kl_feature_mse"

        self.attention_distill_lambda_a = 1.0

        self.attention_distill_lambda_h = 1.0

        self.dropout_rate=0.0

        self.transformer_norm=False

class ppoconfig:
    def __init__(self) -> None:
        self.ori_log_dir = '/tmp/'

        self.num_env_steps=10e6

        self.num_evals = 2

        self.lr = 2.5e-4

        self.log_interval=10

        self.eval_interval=5

        self.max_grad_norm = 0.5

        self.use_linear_lr_decay= True

        self.ACTION_STD_FIXED = True
        
        self.ACTION_STD = 0.2
       
        # Discount factor for rewards
        self.GAMMA = 0.99

        # GAE lambda parameter
        self.GAE_LAMBDA = 0.95

        # Hyperparameter which roughly says how far away the new policy is allowed to
        # go from the old
        self.CLIP_EPS = 0.1

        # Number of epochs (K in PPO paper) of sgd on rollouts in buffer
        self.EPOCHS = 4#8

        # Batch size for sgd (M in PPO paper)
        self.NUM_MINI_BATCH_SIZE =4 #8

        # Value (critic) loss term coefficient
        self.VALUE_COEF = 0.5

        # If KL divergence between old and new policy exceeds KL_TARGET_COEF * 0.01
        # stop updates. Default value is high so that it's not used by default.
        self.KL_TARGET_COEF = 20.0

        # Clip value function
        self.USE_CLIP_VALUE_FUNC = True

        # Entropy term coefficient
        self.ENTROPY_COEF = 0.01

        # Max timesteps per rollout
        self.TIMESTEPS = 128#512
        
        # EPS for Adam/RMSProp
        self.EPS = 1e-5
