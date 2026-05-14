import kagglehub

def main():
    print("Hello from petfinder-my-ai-kohol!")


def loadData():
    print("Loading data...")

    # Download latest version
    path = kagglehub.competition_download('petfinder-adoption-prediction')

    print("Path to competition files:", path)


if __name__ == "__main__":
    main()
    loadData()
