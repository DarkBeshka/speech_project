"""
Скрипт для запуска TensorBoard и анализа attention plots из логов обучения GlowTTS.

Использование:
    python view_tensorboard.py [путь_к_эксперименту]

Если путь не указан, будет использован последний эксперимент.
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

# Путь к директории с экспериментами
EXPERIMENTS_DIR = Path("ruslan_glowtts_exp")

def find_latest_experiment():
    """Находит последний эксперимент по времени модификации."""
    if not EXPERIMENTS_DIR.exists():
        print(f"❌ Директория {EXPERIMENTS_DIR} не найдена!")
        return None
    
    experiments = list(EXPERIMENTS_DIR.glob("run-*"))
    if not experiments:
        print(f"❌ Эксперименты не найдены в {EXPERIMENTS_DIR}")
        return None
    
    # Сортируем по времени модификации (последний = самый свежий)
    latest = max(experiments, key=lambda p: p.stat().st_mtime)
    return latest

def check_tensorboard_logs(exp_path):
    """Проверяет наличие логов TensorBoard в эксперименте."""
    log_files = list(exp_path.glob("events.out.tfevents.*"))
    return len(log_files) > 0, log_files

def analyze_training_log(exp_path):
    """Анализирует текстовый лог обучения для поиска проблем."""
    log_file = exp_path / "trainer_0_log.txt"
    if not log_file.exists():
        return None
    
    print("\n" + "="*70)
    print("📊 АНАЛИЗ ЛОГОВ ОБУЧЕНИЯ")
    print("="*70)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Ищем финальные значения loss
    final_losses = []
    for line in reversed(lines[-500:]):  # Последние 500 строк
        if "loss:" in line and "avg_loss:" not in line:
            try:
                # Пытаемся извлечь значение loss
                parts = line.split("loss:")
                if len(parts) > 1:
                    loss_val = parts[1].strip().split()[0]
                    final_losses.append(float(loss_val))
                    if len(final_losses) >= 5:
                        break
            except:
                pass
    
    if final_losses:
        avg_final_loss = sum(final_losses) / len(final_losses)
        print(f"\n📉 Финальный loss (последние шаги): {avg_final_loss:.4f}")
        
        if avg_final_loss < 0:
            print("✅ Loss отрицательный! Это отличный знак — модель хорошо выучила распределение.")
            print("   (В GlowTTS loss = log_mle + loss_dur, где log_mle может быть отрицательным)")
        elif avg_final_loss > 1.0:
            print("⚠️  ВНИМАНИЕ: Loss всё ещё высокий (>1.0). Модель может быть недообучена.")
        elif avg_final_loss > 0.5:
            print("⚠️  Loss умеренный (0.5-1.0). Модель может нуждаться в дополнительном обучении.")
        else:
            print("✅ Loss низкий (<0.5). Модель должна работать хорошо.")
    
    # Ищем проблемы с градиентами
    grad_norms = []
    for line in reversed(lines[-1000:]):
        if "grad_norm:" in line:
            try:
                parts = line.split("grad_norm:")
                if len(parts) > 1:
                    grad_val = parts[1].strip().split()[0]
                    grad_norms.append(float(grad_val))
                    if len(grad_norms) >= 10:
                        break
            except:
                pass
    
    if grad_norms:
        avg_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)
        print(f"\n📈 Градиенты: средний={avg_grad:.2f}, минимум={min_grad:.2f}, максимум={max_grad:.2f}")
        
        if max_grad > 1000:
            print("🚨 КРИТИЧНО: Экстремально большие градиенты (>1000)! Взрыв градиентов!")
            print("   РЕШЕНИЕ: Уменьшите learning rate до 0.0001, grad_clip до 0.5")
        elif max_grad > 500:
            print("⚠️  КРИТИЧНО: Очень большие градиенты (>500). Серьезный взрыв градиентов!")
            print("   РЕШЕНИЕ: Уменьшите learning rate до 0.0002, grad_clip до 1.0")
        elif max_grad > 100:
            print("⚠️  ВНИМАНИЕ: Большие градиенты (>100). Возможен взрыв градиентов!")
            print("   РЕШЕНИЕ: Уменьшите learning rate, используйте gradient clipping")
        elif avg_grad > 50:
            print("⚠️  Высокие градиенты (средний >50). Может потребоваться уменьшить learning rate.")
        elif avg_grad < 0.1:
            print("⚠️  Очень маленькие градиенты (<0.1). Возможно исчезновение градиентов.")
            print("   РЕШЕНИЕ: Увеличьте learning rate или проверьте архитектуру")
        else:
            print("✅ Градиенты в нормальном диапазоне (1-50). Обучение стабильно.")
    
    # Проверяем наличие ошибок
    error_count = sum(1 for line in lines if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower())
    if error_count > 0:
        print(f"\n❌ Найдено {error_count} упоминаний ошибок в логах!")
    
    return {
        'final_loss': avg_final_loss if final_losses else None,
        'grad_norm': avg_grad if grad_norms else None,
        'errors': error_count
    }

def main():
    # Определяем путь к эксперименту
    if len(sys.argv) > 1:
        exp_path = Path(sys.argv[1])
        if not exp_path.exists():
            print(f"❌ Путь {exp_path} не существует!")
            return
    else:
        exp_path = find_latest_experiment()
        if not exp_path:
            return
    
    print(f"\n🔍 Анализ эксперимента: {exp_path.name}")
    print(f"📁 Полный путь: {exp_path.absolute()}")
    
    # Проверяем наличие логов TensorBoard
    has_logs, log_files = check_tensorboard_logs(exp_path)
    
    if not has_logs:
        print("\n⚠️  Логи TensorBoard не найдены в этом эксперименте.")
    else:
        print(f"\n✅ Найдено {len(log_files)} файл(ов) логов TensorBoard")
    
    # Анализируем текстовый лог
    analysis = analyze_training_log(exp_path)
    
    # Запускаем TensorBoard
    print("\n" + "="*70)
    print("🚀 ЗАПУСК TENSORBOARD")
    print("="*70)
    
    logdir = str(exp_path.absolute())
    print(f"\n📂 Logdir: {logdir}")
    print("\n📝 ИНСТРУКЦИИ ПО ПРОСМОТРУ ATTENTION PLOTS:")
    print("   1. TensorBoard откроется в браузере автоматически")
    print("   2. Перейдите на вкладку 'IMAGES' или 'SCALARS'")
    print("   3. Найдите секцию 'attention' или 'alignment'")
    print("   4. Проверьте выравнивания:")
    print("      ✅ ХОРОШО: Четкая диагональная линия от начала к концу")
    print("      ❌ ПЛОХО: Хаотичные паттерны, размытые или отсутствующие выравнивания")
    print("\n   5. Если выравнивания плохие:")
    print("      - Увеличьте количество эпох обучения")
    print("      - Проверьте качество данных (sample_rate, длительность)")
    print("      - Попробуйте уменьшить learning rate")
    print("      - Увеличьте warmup_steps в lr_scheduler")
    
    print(f"\n🌐 Запускаю TensorBoard...")
    print(f"   Команда: tensorboard --logdir={logdir}")
    print(f"\n   После запуска откройте: http://localhost:6006")
    print(f"   Нажмите Ctrl+C для остановки TensorBoard\n")
    
    try:
        # Use TensorBoard's Python API to start the server programmatically.
        from tensorboard import program

        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", logdir, "--port", "6006"])
        url = tb.launch()
        print(f"\n🚀 TensorBoard запущен: {url}")
        print("Нажмите Ctrl+C в этом окне для остановки TensorBoard.")

        # Keep the script alive while TensorBoard runs until interrupted.
        try:
            while True:
                # Sleep in small increments so KeyboardInterrupt is responsive.
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 TensorBoard остановлен.")
    except ImportError:
        # If tensorboard isn't importable, suggest installation or run via system entrypoint.
        print("\n❌ TensorBoard не установлен как модуль Python.")
        print("   Попробуйте установить: pip install tensorboard")
        # Try falling back to system 'tensorboard' executable if available.
        try:
            subprocess.run(["tensorboard", "--logdir", logdir, "--port", "6006"])
        except FileNotFoundError:
            print("\n❌ Системная команда 'tensorboard' также не найдена.")
    except Exception as e:
        print(f"\n❌ Ошибка при запуске TensorBoard: {e}")

if __name__ == "__main__":
    main()

