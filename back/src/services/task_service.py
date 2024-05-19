from datetime import datetime
import shutil
import uuid
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import sys
sys.path.append('../')
from back.src.schemas.task import TaskCreate, TaskRead
from back.src.services.user_service import get_user_by_email
from back.src.models.task import Task as TaskModel
from worker.demo import classify_image

def get_task_by_id(db: Session, task_id: str) -> TaskRead:
    task = db.query(TaskModel).filter(TaskModel.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

def create_task(db: Session, task: TaskCreate) -> TaskRead:

    if not task.name:
        raise HTTPException(status_code= 404, detail="Task name must be provided")
    
    if not task.user_email:
        raise HTTPException(status_code= 404, detail="User email must be provided")
    user = get_user_by_email(db, task.user_email)
    if not user:
        raise HTTPException(status_code= 404, detail="User email does not exist")
    print(task.input_path.split("/")[-1])
    disease_prediction, disease_probability = classify_image(task.input_path.split("/")[-1])
    
    
    new_task = TaskModel(
        id= task.input_path.split("/")[-1].split(".")[0],
        name = task.name,
        time_stamp = datetime.now(),
        user_email = task.user_email,
        prediction = disease_prediction,
        probability = disease_probability,
        input_path = task.input_path
    )

    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    return new_task


def delete_task(db: Session, task_id: str) -> TaskRead:
    task = get_task_by_id(db, task_id)
    os.remove("uploads/" + task.input_path)
    os.remove("uploads_reason/" + task.input_path)
    db.delete(task)
    db.commit()


