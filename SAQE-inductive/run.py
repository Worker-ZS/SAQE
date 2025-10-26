import yaml


with open('config/complex_query/saqe_main.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("hidden_dims:")
print(config['task']['model']['model']['hidden_dims'])
print(f"length: {len(config['task']['model']['model']['hidden_dims'])}")