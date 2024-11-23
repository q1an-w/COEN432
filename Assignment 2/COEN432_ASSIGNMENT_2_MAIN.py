# Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)
from Base_COEN432_Assignment_2 import main as base_assignment_main
from Bonus_COEN432_Assignment_2 import main as bonus_portion_main


def display_menu():
    """
    Displays the menu options for the user.
    """
    print("Select an option:")
    print("1. Run Base Assignment (K-NN)")
    print("2. Run Bonus Portion (K-NN & Decision Tree w/ Optimization)")
    print("3. Exit")


def main():
    """
    Main function to handle menu navigation and execution.
    """
    while True:
        display_menu()
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                print("\nRunning Base Assignment...\n")
                base_assignment_main()
                print("\n&&&&&&&&&&&&&&&&&&&&&&\n")
            elif choice == 2:
                print("\nRunning Bonus Portion...\n")
                bonus_portion_main()
                print("\n&&&&&&&&&&&&&&&&&&&&&&\n")
            elif choice == 3:
                print("Exiting program ~~~ Byeeeee")
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()
