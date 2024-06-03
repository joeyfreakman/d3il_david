python run.py --config-name=aligning_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=cvae_agent \
              agent_name=cvae \
              window_size=1 \
              group=aligning_cvae_seeds \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=8 \
              agents.model.encoder.latent_dim=16 \
              agents.kl_loss_factor=86.03859051326566