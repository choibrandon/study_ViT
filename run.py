import subprocess

def main():
    config_path = 'config/config.yaml'

    train_command = [
        'python', 'train.py',
        '--config', config_path
    ]

    test_command = [
        'python', 'test.py',
        '--config', config_path
    ]

    print("Starting training...")
    subprocess.run(train_command)

    print("Starting testing...")
    subprocess.run(test_command)

if __name__ == '__main__':
    main()