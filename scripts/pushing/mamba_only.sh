python run.py --config-name=pushing_config \
              --multirun agents=ddpm_mamba_only_agent \
              agent_name=ddpm_mamba_only \
              group=pushing_ddpm_mamba_only_seeds \
              mamba_ssm_cfg.d_state=8,16 \
              mamba_ssm_cfg.d_conv=2,4 \
              seed=0,1,2,3,4,5