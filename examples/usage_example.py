import numpy as np
from simple_strategy import SimpleRLStrategy
from simple_validator import SimpleValidator
from simple_deployer import SimpleDeployer

def main():
    """
    Example usage of the DeepChain framework.
    """
    # 1. Create and train a strategy
    print("1. Creating and training strategy...")
    strategy = SimpleRLStrategy(input_dim=10, hidden_dim=64, output_dim=3)
    
    # Generate some dummy training data
    train_data = {
        'states': np.random.randn(1000, 10),
        'actions': np.random.randint(0, 3, 1000),
        'rewards': np.random.randn(1000)
    }
    
    # Train the strategy
    strategy.train(train_data, n_epochs=10, lr=0.001)
    
    # 2. Validate strategy on blockchain
    print("\n2. Validating strategy on blockchain...")
    validator = SimpleValidator(web3_provider="http://localhost:8545")
    
    # Register strategy
    strategy_id = "example_strategy_001"
    metadata = {
        'name': 'Simple RL Strategy',
        'version': '0.1.0',
        'type': 'reinforcement_learning'
    }
    
    tx_hash = validator.register_strategy(strategy_id, metadata)
    print(f"Strategy registered with hash: {tx_hash}")
    
    # Generate and verify proof
    execution_data = {
        'timestamp': 1234567890,
        'actions': ['buy', 'hold', 'sell'],
        'performance': 0.15
    }
    
    proof = validator.generate_proof(strategy_id, execution_data)
    is_valid = validator.verify_proof(proof)
    print(f"Strategy proof verification: {is_valid}")
    
    # 3. Deploy strategy to edge device
    print("\n3. Deploying strategy to edge...")
    deployer = SimpleDeployer()
    
    # Compress model
    compressed_model = deployer.compress_model(
        strategy.policy,
        target_size=1024 * 1024,  # 1MB
        target_latency=0.1  # 100ms
    )
    
    # Export model
    export_path = "exported_model.onnx"
    deployer.export_model(compressed_model, "onnx", export_path)
    print(f"Model exported to: {export_path}")
    
    # Validate performance
    test_data = {
        'inputs': np.random.randn(100, 10),
        'targets': np.random.randn(100, 3)
    }
    
    performance = deployer.validate_performance(compressed_model, test_data)
    print("\nModel performance metrics:")
    for metric, value in performance.items():
        print(f"{metric}: {value}")
    
    # 4. Make predictions
    print("\n4. Making predictions...")
    current_state = {
        'price': 100.0,
        'volume': 1000000,
        'high': 105.0,
        'low': 95.0,
        'ma_5': 101.0,
        'ma_10': 100.5,
        'rsi': 55.0,
        'macd': 0.5,
        'bollinger_upper': 110.0,
        'bollinger_lower': 90.0
    }
    
    prediction = strategy.predict(current_state)
    print(f"Trading decision: {prediction}")

if __name__ == "__main__":
    main() 