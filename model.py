from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin # Provides default implementations for Flask-Login user methods
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

# Initialize SQLAlchemy (db object will be passed from app.py)
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False) # Increased length for hash
    # Relationships: A user can have many saved recipes and snapshots
    saved_recipes = db.relationship('SavedRecipe', backref='user', lazy=True, cascade="all, delete-orphan")
    scan_snapshots = db.relationship('ScanSnapshot', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """Hashes the password and stores it."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class SavedRecipe(db.Model):
    """Model to store saved recipe details."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    spoonacular_id = db.Column(db.Integer, nullable=False) # Original recipe ID from API
    title = db.Column(db.String(200), nullable=False)
    image_url = db.Column(db.String(500)) # URL for recipe image from API
    recipe_url = db.Column(db.String(500)) # URL for recipe details on Spoonacular
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    # Relationship: A saved recipe has one associated snapshot
    snapshot = db.relationship('ScanSnapshot', backref='saved_recipe', uselist=False, lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<SavedRecipe {self.title} (User: {self.user_id})>'

class ScanSnapshot(db.Model):
    """Model to store information about the saved scan snapshot."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    saved_recipe_id = db.Column(db.Integer, db.ForeignKey('saved_recipe.id'), nullable=False)
    # Store only the filename, assuming it's saved in a specific directory
    filename = db.Column(db.String(100), nullable=False, unique=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def get_url(self, app):
        """Helper function to get the URL for serving this snapshot."""
        # Requires the 'serve_snapshot' endpoint to be defined in app.py
        # and UPLOAD_FOLDER configuration
        upload_folder = app.config.get('UPLOAD_FOLDER', 'user_uploads/snapshots') # Get upload folder from app config
        # This is simplified, ideally use url_for('serve_snapshot', filename=self.filename)
        # but that requires app context here. Returning relative path for now.
        return os.path.join(upload_folder, self.filename).replace("\\", "/") # Ensure forward slashes

    def get_full_path(self, app):
        """Helper function to get the full filesystem path."""
        instance_folder = app.instance_path
        upload_folder_name = app.config.get('UPLOAD_FOLDER_NAME', 'snapshots') # e.g., 'snapshots'
        base_upload_dir = app.config.get('BASE_UPLOAD_DIR', 'user_uploads') # e.g., 'user_uploads'
        return os.path.join(instance_folder, base_upload_dir, upload_folder_name, self.filename)


    def __repr__(self):
        return f'<ScanSnapshot {self.filename} (Recipe: {self.saved_recipe_id})>'

