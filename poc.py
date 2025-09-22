import cv2
import os
import yaml
from ultralytics import YOLO
from ocsort import ocsort
import numpy as np
import time
import requests
from datetime import datetime
import threading

### 1. CARREGAMENTO DE CONFIGURAÇÕES ###
with open('data.yaml', 'r', encoding='utf-8') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

arquivo = data.get('arquivo')
roi_points = data.get('roi_points')
endpoint = data.get('endpoint')
api_key = data.get('api_key')
classes = data.get('classes', [])
camera_name = data.get('camera_name', 'default_cam')
data_yaml = data.get('data')

if not api_key:
    print("ERRO CRÍTICO: A chave de API ('api_key') não foi encontrada no data.yaml. Abortando processo.")
    exit()

### 2. INICIALIZAÇÃO DE MODELOS E VARIÁVEIS ###
model = YOLO("yolov10m.pt") 
# model.to('cuda') # Descomente se tiver uma GPU NVIDIA com CUDA configurado
roi = tuple(roi_points)

# Cores para visualização
colors = {
    "person": (255, 0, 0),
    "bicycle": (0, 255, 0),
    "car": (0, 0, 255),
    "motorcycle": (0, 123, 255),
    "bus": (44, 99, 255),
    "truck": (128, 196, 255),
}

# Dicionários e controle de lote
tracked_trajectories = {}
completed_trajectories_buffer = {}
previous_frame_ids = set()
BATCH_SEND_THRESHOLD = 100

# Parâmetros de controle de trajetória
MIN_DISTANCE_THRESHOLD = 2.0 # Distância mínima (pixels) para salvar um novo ponto
MIN_DIRECTION_CHANGE_COSINE = 0.98 # Cosseno do ângulo mínimo de mudança de direção

# Inicialização do Tracker
tracker_cfg = data.get('tracker_params', {})
tracker = ocsort.OCSort(
    det_thresh=tracker_cfg.get('det_thresh', 0.4),
    max_age=tracker_cfg.get('max_age', 30),
    min_hits=tracker_cfg.get('min_hits', 3),
    iou_threshold=tracker_cfg.get('iou_threshold', 0.3),
    use_byte=tracker_cfg.get('use_byte', False)
)

# Captura de vídeo
cap = cv2.VideoCapture(arquivo)

# Processamento da data base e tempo de início
datetime_obj = None
if data_yaml and isinstance(data_yaml, list) and len(data_yaml) > 0:
    data_arq = data_yaml[0]
    if isinstance(data_arq, str):
        try:
            datetime_obj = datetime.strptime(data_arq, "%d-%m-%Y %H:%M:%S")
            print(f"Usando data e hora base do YAML: {datetime_obj.strftime('%Y-%m-%d %H:%M:%S')}")
        except ValueError:
            datetime_obj = None
if datetime_obj is None:
    datetime_obj = datetime.now()
    print(f"AVISO: Data do YAML inválida ou não encontrada. Usando data e hora atuais como base: {datetime_obj.strftime('%Y-%m-%d %H:%M:%S')}")

unix_timestamp = datetime_obj.timestamp()
processing_start_time = time.time()


### 3. FUNÇÕES AUXILIARES ###
def calculate_iou(box1, box2):
    """Calcula a Interseção sobre União (IoU) de duas caixas delimitadoras."""
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Função para selecionar os pontos mais importantes de uma trajetória, priorizando aqueles com maior curvatura
def select_best_points(points, target_count=50):
    """ Seleciona um número alvo de pontos de uma trajetória, priorizando os de maior curvatura. """
    if len(points) <= target_count:
        return points

    points_np = np.array(points, dtype=np.float32)
    intermediate_points = points_np[1:-1]
    
    vec_prev = intermediate_points - points_np[:-2]
    vec_next = points_np[2:] - intermediate_points
    
    norm_prev = np.linalg.norm(vec_prev, axis=1)
    norm_next = np.linalg.norm(vec_next, axis=1)
    
    safe_indices = np.where((norm_prev > 1e-6) & (norm_next > 1e-6))[0]
    
    importance_scores = np.zeros(len(intermediate_points))
    if len(safe_indices) > 0:
        dot_product = np.einsum('ij,ij->i', vec_prev[safe_indices], vec_next[safe_indices])
        cos_angle = np.clip(dot_product / (norm_prev[safe_indices] * norm_next[safe_indices]), -1.0, 1.0)
        importance_scores[safe_indices] = 1.0 - cos_angle
    
    num_to_keep = target_count - 2
    indices_to_keep = np.argsort(importance_scores)[-num_to_keep:]
    
    final_indices = {0, len(points) - 1}
    for idx in indices_to_keep:
        final_indices.add(idx + 1)
        
    sorted_indices = sorted(list(final_indices))
    
    return [points[i] for i in sorted_indices]

def gerar_e_enviar_arquivo_final(trajetorias_completas, url_webservice, camera_id, api_key):
    """Gera um arquivo .txt no formato esperado e faz o upload para o webservice."""
    if not trajetorias_completas:
        print("Nenhuma trajetória no buffer para enviar.")
        return
    nome_arquivo = f"trajetorias_data_{int(time.time())}.txt"
    print(f"Gerando arquivo de trajetória: {nome_arquivo}...")
    try:
        with open(nome_arquivo, "w") as f:
            for obj_id, data in trajetorias_completas.items():
                id_obj = data.get('id', obj_id)
                id_classe_obj = data.get('class_id', -1)
                ts_inicio = data.get('start_time', 0)
                ts_fim = data.get('last_seen_time', 0)
                pontos = data.get('points', [])
                linha = f"{id_obj}, {id_classe_obj}, {ts_inicio}, {ts_fim}, {pontos}\n"
                f.write(linha)
        print(f"Arquivo gerado. Enviando para {url_webservice}...")
        
        # Usa o api_key para autenticação da requisição
        headers = {'X-API-KEY': api_key}
        payload_data = {'camera': camera_id}
        
        # Preparar o arquivo para upload
        with open(nome_arquivo, 'rb') as f_upload:
            files = {'file': (nome_arquivo, f_upload, 'text/plain')}
            response = requests.post(url_webservice, headers=headers, files=files, data=payload_data, timeout=30)
            if 200 <= response.status_code < 300:
                print(f"Lote de {len(trajetorias_completas)} trajetórias enviado com sucesso!")
            else:
                print(f"Erro no envio do lote. Status: {response.status_code}, Resposta: {response.text}")
    except Exception as e:
        print(f"Ocorreu uma exceção no envio: {e}")
    finally:
        if os.path.exists(nome_arquivo):
            os.remove(nome_arquivo)


### 4. LAÇO PRINCIPAL DE PROCESSAMENTO ###
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro na captura.")
        break

    x, y, w, h = roi[0]
    roi_frame = frame[y:y+h, x:x+w]

    results = model.predict(source=roi_frame, save=False, show=False, conf=0.5, classes=classes, imgsz=320, verbose=False)

    yolo_detections = []
    dets_for_tracker = []
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            yolo_detections.append({'bbox': [x1, y1, x2, y2], 'class_id': int(cls)})
            dets_for_tracker.append([x1, y1, x2, y2, conf, int(cls)])

    track_results = tracker.update(np.array(dets_for_tracker), None) if dets_for_tracker else tracker.update(np.empty((0, 6)), None)

    current_frame_ids = set()
    for result_track in track_results:
        x1_track, y1_track, x2_track, y2_track, tracker_id = result_track[:5]
        tracker_id = int(tracker_id)
        current_frame_ids.add(tracker_id)
        track_bbox = [int(x1_track), int(y1_track), int(x2_track), int(y2_track)]
        
        centroid_x = x + (track_bbox[0] + track_bbox[2]) // 2
        centroid_y = y + (track_bbox[1] + track_bbox[3]) // 2
        new_point = (centroid_x, centroid_y)
        
        current_real_time = time.time()
        elapsed_processing_time = current_real_time - processing_start_time
        timestamp = unix_timestamp + elapsed_processing_time

        if tracker_id in tracked_trajectories:
            points = tracked_trajectories[tracker_id]['points']
            last_point = points[-1]
            distance = np.linalg.norm(np.array(new_point) - np.array(last_point))
            if distance < MIN_DISTANCE_THRESHOLD:
                tracked_trajectories[tracker_id]['last_seen_time'] = timestamp
            else:
                add_point = True
                if len(points) >= 2:
                    vec_last = np.array(points[-1]) - np.array(points[-2])
                    vec_new = np.array(new_point) - np.array(points[-1])
                    norm_last, norm_new = np.linalg.norm(vec_last), np.linalg.norm(vec_new)
                    if norm_last > 0 and norm_new > 0:
                        cos_angle = np.dot(vec_last, vec_new) / (norm_last * norm_new)
                        if cos_angle > MIN_DIRECTION_CHANGE_COSINE:
                            points[-1] = new_point
                            add_point = False
                if add_point:
                    points.append(new_point)
                tracked_trajectories[tracker_id]['last_seen_time'] = timestamp
        else:
            best_iou, best_class_id = 0.0, -1
            for det in yolo_detections:
                iou = calculate_iou(track_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_class_id = iou, det['class_id']
            if best_class_id != -1 and best_iou > 0.3:
                tracked_trajectories[tracker_id] = {
                    'id': tracker_id,
                    'class': model.names[best_class_id],
                    'class_id': best_class_id,
                    'start_time': timestamp,
                    'last_seen_time': timestamp,
                    'points': [new_point]
                }
    
    # Desenhar Bounding Boxes e Rastros dos objetos ativos
    for obj_data in tracked_trajectories.values():
        for res in track_results:
            if int(res[4]) == obj_data['id']:
                x1, y1, x2, y2 = map(int, res[:4])
                label = obj_data['class']
                box_color = colors.get(label.lower(), (200, 200, 200))
                cv2.rectangle(frame, (x + x1, y + y1), (x + x2, y + y2), box_color, 2)
                cv2.putText(frame, f"ID:{obj_data['id']} {label}", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                break
        points_to_draw = obj_data.get('points', [])
        obj_color = colors.get(obj_data.get('class', '').lower(), (200, 200, 200))
        for i, point in enumerate(points_to_draw):
            radius = int(((i + 1) / len(points_to_draw)) * 4) + 1
            cv2.circle(frame, point, radius, obj_color, -1)

    # Detectar objetos que saíram, aplicar seleção de pontos e mover para o buffer
    disappeared_ids = previous_frame_ids - current_frame_ids
    for d_id in disappeared_ids:
        if d_id in tracked_trajectories:
            full_trajectory = tracked_trajectories[d_id]
            full_points = full_trajectory['points']
            
            # Aplicar seleção de pontos para trajetórias longas
            if len(full_points) >= 50:
                print(f"Trajetória ID {d_id} ({full_trajectory['class']}) é longa ({len(full_points)} pontos). Selecionando os melhores 50...")
                selected_points = select_best_points(full_points, target_count=50)
                full_trajectory['points'] = selected_points
                print(f"Trajetória ID {d_id} reduzida para {len(selected_points)} pontos.")
            
            completed_trajectories_buffer[d_id] = full_trajectory
            del tracked_trajectories[d_id]
    previous_frame_ids = current_frame_ids.copy()

    # Enviar lote se o buffer atingir o limite
    if len(completed_trajectories_buffer) >= BATCH_SEND_THRESHOLD:
        print(f"Buffer atingiu {len(completed_trajectories_buffer)} trajetórias. Iniciando envio do lote...")
        buffer_para_envio = completed_trajectories_buffer.copy()
        completed_trajectories_buffer.clear()
        threading.Thread(target=gerar_e_enviar_arquivo_final, args=(buffer_para_envio, endpoint, camera_name, api_key)).start()

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Processamento interrompido pelo usuário.")
        break

### 5. FINALIZAÇÃO E ENVIO DO ARQUIVO ###
print("Finalizando o processamento do vídeo.")
print(f"Movendo {len(tracked_trajectories)} trajetórias ativas restantes para o buffer final.")
for final_id, final_data in tracked_trajectories.items():
    final_data['last_seen_time'] = unix_timestamp + (time.time() - processing_start_time)
    
    # Seleção de pontos para trajetórias longas
    full_points = final_data['points']
    if len(full_points) >= 50:
        print(f"Trajetória final ID {final_id} ({final_data['class']}) é longa ({len(full_points)} pontos). Selecionando os melhores 50...")
        selected_points = select_best_points(full_points, target_count=50)
        final_data['points'] = selected_points
        print(f"Trajetória final ID {final_id} reduzida para {len(selected_points)} pontos.")
        
    completed_trajectories_buffer[final_id] = final_data

if completed_trajectories_buffer:
    print(f"Enviando lote final de {len(completed_trajectories_buffer)} trajetórias...")
    thread_final = threading.Thread(target=gerar_e_enviar_arquivo_final, args=(completed_trajectories_buffer.copy(), endpoint, camera_name, api_key))
    thread_final.start()
    thread_final.join()
    print("Envio final concluído.")
else:
    print("Nenhuma trajetória no buffer final para enviar.")

cap.release()
cv2.destroyAllWindows()
print("Script finalizado.")
