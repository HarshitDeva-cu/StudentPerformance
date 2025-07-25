# 🎓 Student Performance Predictor

A machine learning web application that predicts whether a student will pass or fail based on various academic and personal factors.

## 📋 Features

- **Interactive Web Interface**: Clean Streamlit UI for easy data input
- **ML-Based Predictions**: Random Forest classifier with ~85-90% accuracy
- **Comprehensive Analysis**: Considers 21 different student attributes
- **Real-time Predictions**: Instant results with confidence scores
- **Actionable Insights**: Provides personalized recommendations

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **Pickle** - Model persistence

## 📁 Project Structure

```
student-predictor/
├── generate_data.py      # Generate synthetic student data
├── train_model.py        # Train the ML model
├── app.py               # Streamlit application
├── model.pkl            # Trained model (generated)
├── student_data.csv     # Dataset (generated)
├── feature_importance.csv # Feature rankings (generated)
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/student-predictor.git
cd student-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Model
```bash
# Generate synthetic student data
python generate_data.py

# Train the model
python train_model.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Dataset Features

The model uses 21 features across 4 categories:

### Demographics
- Gender, Age
- Parent Education Level
- Parent Income
- School Type

### Academic Performance
- Study Hours
- Attendance Percentage
- Internal & Final Marks
- Extra Classes
- Test Preparation

### Behavioral Factors
- Peer Group Study
- Screen Time
- Library Usage
- Sports Participation
- Discipline Score

### Well-being
- Sleep Hours
- Mental Stress Level
- Learning Disability
- Family Support

## 🤖 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~85-90%
- **Train/Test Split**: 80/20
- **Key Features**: Attendance, Final Marks, Study Hours

## 🌐 Deployment

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Environment Variables (if needed)
```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 📝 Usage Example

1. Open the web application
2. Fill in student details:
   - Demographics information
   - Academic performance metrics
   - Behavioral and well-being factors
3. Click "Predict Performance"
4. View results with confidence score
5. Read personalized recommendations

## 🔧 Customization

### Modify Model Parameters
Edit `train_model.py`:
```python
model = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Tree depth
    random_state=42
)
```

### Add New Features
1. Update `generate_data.py` with new columns
2. Modify `train_model.py` preprocessing
3. Add input fields in `app.py`

## 📈 Future Enhancements

- [ ] Add more ML algorithms (SVM, XGBoost)
- [ ] Implement cross-validation
- [ ] Add data visualization dashboard
- [ ] Enable CSV upload for batch predictions
- [ ] Add API endpoint for predictions
- [ ] Implement user authentication
- [ ] Store prediction history

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn community
- Contributors and testers

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/your-username/student-predictor](https://github.com/your-username/student-predictor)