# Vision-RAG Web Application  

**Vision-RAG** is an advanced web application designed to streamline peer benchmarking by utilizing pre-determined KPI questions. The application automates the generation of benchmarking sheets and provides an easy-to-use interface for users to interact with.  

---

## Features  

1. **Pre-Defined KPI Questions**
   - Set pre-determined KPI questions for benchmarking.
   - Customizable to fit specific projects and use cases.

2. **Automated Peer Benchmarking**  
   - Automatically generates a peer benchmarking sheet based on the defined KPIs.  
   - Efficiently answers all questions using robust, pre-trained models.  

3. **Seamless Report Download**  
   - Download the report with a single click once the benchmarking sheet is ready.  
   - Reports are structured for clarity and easy interpretation.  

---

## How It Works  

1. **Set KPI Questions**  
   - Input and configure the KPI questions within the application.
     ![image](https://github.com/user-attachments/assets/1b00f674-0f61-45cc-b97c-350e04525d8e)


2. **Create a Sector**
   - Create a sector with a short description.
     ![image](https://github.com/user-attachments/assets/3d765647-ad5e-4e66-a1a0-8cae02ddde18)


4. **Create a Project**  
   - Start a new project within a sector and upload the necessary files (PDFs and XML) and year.
     ![image](https://github.com/user-attachments/assets/180ac2ed-dacf-4fb6-87f1-f447c5624a75)
     ![image](https://github.com/user-attachments/assets/07f13fa0-a844-494b-8c61-31b2204f3503)


  
5. **Check Project Status**
   - Check if the model has finished processing your newly created project or not.
     ![image](https://github.com/user-attachments/assets/3c1c2f80-cd05-4133-8722-b86efe6c44cf)
     ![image](https://github.com/user-attachments/assets/a3b9bcb5-0609-4b83-bf5e-1e5bf52097d2)



6. **Peer Benchmarking Report Generation**
   - You can load the project and generate the responses for the predetermined KPIs or use the chat option to query any custom question.

     ![image](https://github.com/user-attachments/assets/5624eb18-1c74-4c57-ab21-06c97b92b872)


7. **Download Report**  
   - Download the finalized peer benchmarking report for your projects with one click.  

---

## URL

```
http://13.127.221.61/
```

---

## Usage  

1. Open the web application in your browser.  
2. Set your KPI questions and create a new project.  
3. Wait for the system to process and generate the benchmarking sheet.  
4. Download your report directly from the dashboard.  

---

## Future Enhancements  

- Addition of sector-specific KPI libraries.  
- Enhanced data visualization within reports.  
- Support for comparing different projects from the same sector together.
- Enhance how batches are handled. If any batch fails for some reason, the model should send another batch request without discarding the other completed results.

---

**Enjoy using Vision-RAG!**  
