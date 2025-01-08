from grid_world import RelayConfig, ClientConfig, InitConfig

size = 1000
relay_config = RelayConfig(num=10, speed=10.0, limit_position=True, limit_height=True, max_height=100.0, min_height=50.0)
client_config = ClientConfig(num=100, speed=5.0, traffic=2.0, link_establish=2.0, is_move=False)
init_config = InitConfig(center_type="center", relay_type="follow_nearby", client_type="random")
max_episode_steps = 500