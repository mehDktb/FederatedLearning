import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
client_metrics_1 = pd.read_csv('client_metrics_1.csv')
client_metrics_2 = pd.read_csv('client_metrics_2.csv')
federated_model = pd.read_csv('federated_training_results.csv')

# Plotting accuracy comparison
plt.figure(figsize=(12, 6))
plt.plot(client_metrics_1['Round'], client_metrics_1['Train Accuracy'], marker='o', linestyle='-', label='Client Metrics 1 Train Accuracy')
plt.plot(client_metrics_2['Round'], client_metrics_2['Train Accuracy'], marker='s', linestyle='-', label='Client Metrics 2 Train Accuracy')
plt.plot(federated_model['Round'], federated_model['Accuracy'], marker='^', linestyle='-', label='Federated Model Accuracy')

plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Train Accuracy Comparison for Clients and Federated Model')
plt.legend()
plt.grid(True)
plt.show()
