from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
from fastapi import UploadFile

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement your auth logic
    if not valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    return user

def validate_csv(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            400, 
            "Only CSV files are supported"
        )
    return file