python run.py --config-name=avoiding_config \
              --multirun agents=ddpm_mamba_only_agent \
              agent_name=ddpm_mamba_only \
              group=avoiding_ddpm_mamba_only_seeds \
              mamba_ssm_cfg.d_state=8,16 \
              mamba_ssm_cfg.d_conv=2,4 \
              seed=0,1,2,3,4,5