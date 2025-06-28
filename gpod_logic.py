import json
import os
import random
from datetime import datetime
import hashlib

class GPODAuth:
    def __init__(self, db, User):
        self.db = db
        self.User = User
        self.challenges = {}  # Keep challenges in memory as they're temporary
        
    def register_user(self, data):
        """
        Register a new user with their object preferences
        """
        username = data.get('username')
        if not username:
            return False
            
        # Check if username already exists
        if self.User.query.filter_by(username=username).first():
            return False
            
        # Hash the username for storage
        username_hash = hashlib.sha256(username.encode()).hexdigest()
        
        # Create new user
        new_user = self.User(
            username=username,
            username_hash=username_hash,
            objects=json.dumps(data.get('objects', [])),
            background_type=data.get('background_type', 'default'),
            is_admin=False  # Default to non-admin
        )
        
        try:
            self.db.session.add(new_user)
            self.db.session.commit()
            return True
        except Exception as e:
            print(f"Error registering user: {str(e)}")
            self.db.session.rollback()
            return False
            
    def get_user_preferences(self, username):
        """
        Get user preferences for challenge generation
        """
        user = self.User.query.filter_by(username=username).first()
        if user:
            return {
                'username': user.username,
                'objects': json.loads(user.objects),
                'background_type': user.background_type
            }
        return None
        
    def verify_login(self, data):
        """
        Verify user login attempt
        """
        username = data.get('username')
        if not username:
            return False
            
        user = self.User.query.filter_by(username=username).first()
        if user:
            return {'success': True, 'is_admin': user.is_admin}
        return {'success': False}
        
    def delete_user(self, username):
        """
        Delete a user (admin only)
        """
        user = self.User.query.filter_by(username=username).first()
        if user:
            try:
                self.db.session.delete(user)
                self.db.session.commit()
                return True
            except Exception as e:
                print(f"Error deleting user: {str(e)}")
                self.db.session.rollback()
                return False
        return False
        
    def get_all_users(self):
        """
        Get all users (admin only)
        """
        users = self.User.query.all()
        return [{
            'username': user.username,
            'objects': json.loads(user.objects),
            'background_type': user.background_type,
            'created_at': user.created_at.isoformat(),
            'is_admin': user.is_admin
        } for user in users]
        
    def generate_challenge(self, background_type, objects):
        """
        Generate a new challenge for authentication
        """
        challenge_id = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()
        
        # Select random objects for the challenge
        selected_objects = random.sample(objects, min(3, len(objects)))
        
        challenge = {
            'id': challenge_id,
            'objects': selected_objects,
            'background_type': background_type,
            'created_at': datetime.now().isoformat()
        }
        
        self.challenges[challenge_id] = challenge
        return challenge
        
    def verify_clicks(self, data):
        """
        Verify if the clicked points match the challenge objects
        """
        challenge_id = data.get('challenge_id')
        clicks = data.get('clicks', [])
        
        if challenge_id not in self.challenges:
            return {'success': False, 'message': 'Invalid challenge'}
            
        challenge = self.challenges[challenge_id]
        expected_objects = set(challenge['objects'])
        
        # Verify each click is on a valid object
        for click in clicks:
            if click['object'] not in expected_objects:
                return {'success': False, 'message': 'Invalid object clicked'}
                
        # Clean up the challenge
        del self.challenges[challenge_id]
        
        return {'success': True, 'message': 'Authentication successful'} 
