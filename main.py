from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_network


def menu():

    while True:

        print("\n" + "=" * 60)
        print(" INTRUSION DETECTION SYSTEM USING LSTM ")
        print("=" * 60)

        print("1. Train Model")
        print("2. Evaluate Model")
        print("3. Predict New Data")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":

            print("\nTraining Model...\n")

            train_model()

        elif choice == "2":

            print("\nEvaluating Model...\n")

            evaluate_model()

        elif choice == "3":

            csv_path = input(
                "\nEnter CSV file path (Example: data/sample.csv): "
            )

            predict_network(csv_path)

        elif choice == "4":

            print("\nThank you for using Intrusion Detection System.")

            break

        else:

            print("\nInvalid Choice! Please try again.")


if __name__ == "__main__":

    menu()