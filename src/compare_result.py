import matplotlib.pyplot as plt

labels = ['Centralized', 'Federated (Clean)', 'Federated (Malicious)']
accuracies = [81.57, 75.48, 68.27]

plt.bar(labels, accuracies)
plt.ylabel('Test Accuracy (%)')
plt.title('Centralized vs Federated vs Federated with Malicious Client')
plt.ylim(0, 100)
plt.savefig('results/result_comparison.png', dpi = 300)
plt.show()
