import jax.numpy as jnp
from config import config
from data.loader import get_mnist_data, get_mnist_test_data
from models.mlp import init_mlp_params, mlp
from utils.training import train
from utils.metrics import accuracy_fn
from utils.optimizer import create_optimizer
from utils.plotting import plot_training_curves

def main():
    # 1. 데이터 로딩
    X_train, y_train, X_val, y_val = get_mnist_data()
    X_test, y_test = get_mnist_test_data()

    # 2. 모델 및 옵티마이저 설정
    config['model'] = mlp
    config['optimizer'] = create_optimizer(config['learning_rate'])

    # 3. 파라미터 초기화
    params = init_mlp_params(config['key'], config['model_arch'])

    # 4. 학습 수행
    params, loss_history, val_acc_history, test_acc_history = train(
        params, X_train, y_train, X_val, y_val, X_test, y_test, config
    )

    # 5. 최종 정확도 출력
    val_acc = accuracy_fn(params, jnp.array(X_val), jnp.array(y_val), config['model'])
    print(f"\n✅ Final Validation Accuracy: {val_acc:.4f}")

    test_acc = accuracy_fn(params, jnp.array(X_test), jnp.array(y_test), config['model'])
    print(f"🧪 Final Test Accuracy: {test_acc:.4f}")

    # 6. 학습 곡선 시각화
    plot_training_curves(
        loss_history,
        val_acc_history,
        test_acc_history,
        save_path="logs/result.png"
    )


if __name__ == "__main__":
    main()
