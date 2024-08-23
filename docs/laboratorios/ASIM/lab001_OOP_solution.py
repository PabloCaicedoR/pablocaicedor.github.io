# Define the Person class
class Person:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender


# Define the Patient class, inheriting from Person
class Patient(Person):
    def __init__(self, name, age, gender, patient_id, illness):
        super().__init__(name, age, gender)
        self.patient_id = patient_id
        self.illness = illness


# Define the Doctor class, inheriting from Person
class Doctor(Person):
    def __init__(self, name, age, gender, doctor_id, specialization):
        super().__init__(name, age, gender)
        self.doctor_id = doctor_id
        self.specialization = specialization


# Define the Appointment class
class Appointment:
    def __init__(self, appointment_id, patient, doctor, date, time):
        self.appointment_id = appointment_id
        self.patient = patient
        self.doctor = doctor
        self.date = date
        self.time = time


# Define the Hospital class
class Hospital:
    def __init__(self):
        self.patients = []
        self.doctors = []
        self.appointments = []

    def add_patient(self, name, age, gender, patient_id, illness):
        patient = Patient(name, age, gender, patient_id, illness)
        self.patients.append(patient)

    def add_doctor(self, name, age, gender, doctor_id, specialization):
        doctor = Doctor(name, age, gender, doctor_id, specialization)
        self.doctors.append(doctor)

    def schedule_appointment(self, appointment_id, patient_id, doctor_id, date, time):
        patient = next((p for p in self.patients if p.patient_id == patient_id), None)
        doctor = next((d for d in self.doctors if d.doctor_id == doctor_id), None)
        if patient and doctor:
            appointment = Appointment(appointment_id, patient, doctor, date, time)
            self.appointments.append(appointment)
        else:
            print("Patient or doctor not found")

    def list_appointments(self):
        for appointment in self.appointments:
            print(f"Appointment ID: {appointment.appointment_id}")
            print(f"Patient: {appointment.patient.name}")
            print(f"Doctor: {appointment.doctor.name}")
            print(f"Date: {appointment.date}")
            print(f"Time: {appointment.time}")
            print("------------------------")


medicos = Doctor("Fulano de Tal", 35, "Male", "D009", "Internist")

# Create a Hospital object
hospital = Hospital()

# Add patients, doctors, and schedule appointments
hospital.add_patient("John Doe", 30, "Male", "P001", "Flu")
hospital.add_doctor("Jane Smith", 35, "Female", "D001", "General Practitioner")
hospital.schedule_appointment("A001", "P001", "D001", "2024-08-20", "10:00 AM")
hospital.schedule_appointment("A001", "P001", "D001", "2024-08-20", "11:00 AM")
hospital.list_appointments()
