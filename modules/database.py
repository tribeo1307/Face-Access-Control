# Database management module
import csv
import os
from datetime import datetime
import pickle 

class DatabaseManager:
    def __init__(self, log_file='logs/access_log.csv', model_dir='models/'):
        self.log_file = log_file
        self.model_dir = model_dir
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'User_ID', 'Status', 'Message'])
            except IOError as e:
                print(f"Lỗi: Không thể tạo tệp log {self.log_file}. {e}")

    def save_model(self, model, model_name):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Mô hình '{model_name}' đã được lưu tại: {model_path}")
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")

    def load_model(self, model_name):
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            return None
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Mô hình '{model_name}' đã được tải.")
            return model
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return None

    def log_access(self, user_id='UNKNOWN', status='FAILED', message=''):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = [timestamp, user_id, status, message]

        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_entry)
        except Exception as e:
            print(f"Lỗi khi ghi log: {e}")
