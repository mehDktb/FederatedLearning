import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
client1_df = pd.read_csv('/home/mehdi_ktb/Documents/FederatedLearning/sand_box/second_try/client_metrics_1.csv')
client2_df = pd.read_csv('/home/mehdi_ktb/Documents/FederatedLearning/sand_box/second_try/client_metrics_2.csv')
fed_df = pd.read_csv('/home/mehdi_ktb/Documents/FederatedLearning/sand_box/second_try/federated_training_results.csv')

# Assuming the CSV files have a "Round" column.
# For the client metrics files, we use the "Test Accuracy" column.
# For the federated training results file, we assume the accuracy column is named "Accuracy".
# If the federated file uses a different column name, adjust accordingly.

plt.figure(figsize=(10, 6))
plt.plot(client1_df['Round'], client1_df['Test Accuracy'], marker='o', label='Client 1 Test Accuracy')
plt.plot(client2_df['Round'], client2_df['Test Accuracy'], marker='o', label='Client 2 Test Accuracy')
plt.plot(fed_df['Round'], fed_df['Accuracy'], marker='o', label='Federated Training Accuracy')

plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.show()
