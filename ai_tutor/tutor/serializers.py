from rest_framework import serializers

class InstructionSerializer(serializers.Serializer):
    instruction = serializers.CharField()
