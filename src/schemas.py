from pydantic import BaseModel,field_validator

class Prompt(BaseModel):
    message:str

    @field_validator("message")
    @classmethod
    def validate(value,cls):
        if type(value)!=str:
            raise TypeError("Must be string")
        return value
