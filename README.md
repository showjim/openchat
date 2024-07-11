## OpenChat – Web APP

### Overview

A simple LLM chatbot Webapp based on Streamlit and

### Getting Started

1. **Pull the Docker Image**:
   ```sh
   docker pull showjimzc/openchat:latest
   ```

2. **Setup OpenRouter**: 
To configure the application, you will need to create a key.txt file containing your OpenRouter API key and a config.json file with your desired settings. ***And put them in "setting" fold***.

3. **Run the Container**:
   ```sh
   docker run -d -p 8501:8501 -v <host setting folder>:/app/setting showjimzc/openchat:latest
   ```

4. **Access the Tool: Open http://localhost:8501 in your browser.**


### Support

For docker, refer to [OpenChat](https://hub.docker.com/r/showjimzc/openchat).