from app import app, db, User
import hashlib
import json

def create_admin_user(username="admin"):
    with app.app_context():
        # Check if admin already exists
        existing_admin = User.query.filter_by(username=username).first()
        if existing_admin:
            print(f"Admin user '{username}' already exists!")
            return
        
        # Create admin user
        admin = User(
            username=username,
            username_hash=hashlib.sha256(username.encode()).hexdigest(),
            objects=json.dumps([]),  # Empty list of objects
            background_type="indoor",  # Default background
            is_admin=True  # Set admin flag
        )
        
        try:
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user '{username}' created successfully!")
        except Exception as e:
            db.session.rollback()
            print(f"Error creating admin user: {str(e)}")

if __name__ == "__main__":
    create_admin_user() 