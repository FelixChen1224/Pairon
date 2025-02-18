# track_info_db.py

import sqlite3
from datetime import datetime
import json
import numpy as np

class TrackInfoDatabase:
    
    def __init__(self, db_path='track_analysis.db'):
        """
        Initialize database connection and create necessary tables
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        
    
    def create_tables(self):
        """Create all necessary database tables if they don't exist"""
        try:
            # Base tracking information with REPLACE support
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS track_base (
                    track_id INTEGER PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_frames INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    appear_image_path TEXT,
                    disappear_image_path TEXT
                )
            ''')
            
            # Feature analysis results
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS track_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    timestamp TEXT NOT NULL,
                    feature_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    feature_value TEXT NOT NULL,
                    confidence REAL,
                    FOREIGN KEY (track_id) REFERENCES track_base (track_id) ON DELETE CASCADE
                )
            ''')
            
            # Style analysis results
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS track_style (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    timestamp TEXT NOT NULL,
                    dominant_colors TEXT,
                    dominant_patterns TEXT,
                    style_categories TEXT,
                    FOREIGN KEY (track_id) REFERENCES track_base (track_id) ON DELETE CASCADE
                )
            ''')
            
            # Position tracking data
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS track_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER,
                    x REAL,
                    y REAL,
                    width REAL,
                    height REAL,
                    confidence REAL,
                    FOREIGN KEY (track_id) REFERENCES track_base (track_id) ON DELETE CASCADE
                )
            ''')
            
            # Enable foreign key support
            self.cursor.execute('PRAGMA foreign_keys = ON')
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error in create_tables: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error in create_tables: {e}")
            raise
    
    def update_track_info(self, track_id, timestamp, is_new=False):
        """
        Update or insert track information
        
        Args:
            track_id: Target tracking ID
            timestamp: Current timestamp
            box_data: Bounding box coordinates [x1, y1, x2, y2]
            is_new: Whether this is a new track
        """
        try:
            # Check if track exists
            self.cursor.execute('''
                SELECT track_id FROM track_base WHERE track_id = ?
            ''', (track_id,))
            exists = self.cursor.fetchone() is not None
            
            if exists:
                # Update existing track
                self.cursor.execute('''
                    UPDATE track_base 
                    SET last_seen = ?,
                        total_frames = total_frames + 1
                    WHERE track_id = ?
                ''', (timestamp, track_id))
            else:
                # Insert new track
                self.cursor.execute('''
                    INSERT INTO track_base 
                    (track_id, first_seen, last_seen, total_frames)
                    VALUES (?, ?, ?, 1)
                ''', (track_id, timestamp, timestamp))
            
           
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error in update_track_info: {e}")
            print(f"Parameters: track_id={track_id}, timestamp={timestamp}, box_data={box_data}, is_new={is_new}")
            self.conn.rollback()
        except Exception as e:
            print(f"Unexpected error in update_track_info: {e}")
            print(f"Parameters: track_id={track_id}, timestamp={timestamp}, box_data={box_data}, is_new={is_new}")
            self.conn.rollback()

    
    def update_track_disappearance(self, track_id, timestamp, face_img_path=None):
        """
        Update track disappearance information
        
        Args:
            track_id: Target tracking ID
            timestamp: Disappearance timestamp
            face_img_path: Path to the last face image
        """
        try:
            self.cursor.execute('''
                UPDATE track_base 
                SET last_seen = ?,
                    disappear_image_path = COALESCE(?, disappear_image_path)
                WHERE track_id = ?
            ''', (timestamp, face_img_path, track_id))
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            self.conn.rollback()
    
    def update_features(self, track_id, feature_matches, style_analysis, top_features):
        """
        Update feature analysis results
        
        Args:
            track_id: Target tracking ID
            feature_matches: Dictionary of feature matches from analyzer
            style_analysis: Style analysis results
            top_features: List of top features
        """
        try:
            # Store individual features if they exist
            if isinstance(feature_matches, dict):
                if 'individual' in feature_matches:
                    for category, features in feature_matches['individual'].items():
                        if isinstance(features, list):
                            for feature in features:
                                self.cursor.execute('''
                                    INSERT INTO track_features 
                                    (track_id, timestamp, feature_type, category, feature_value, confidence)
                                    VALUES (?, datetime('now'), ?, ?, ?, ?)
                                ''', (
                                    track_id, 
                                    'individual', 
                                    category,
                                    feature.get('feature', ''),
                                    feature.get('confidence', 0.0)
                                ))
                
                # Store combined features if they exist
                if 'combined' in feature_matches:
                    for category, features in feature_matches['combined'].items():
                        if isinstance(features, list):
                            for feature in features:
                                self.cursor.execute('''
                                    INSERT INTO track_features 
                                    (track_id, timestamp, feature_type, category, feature_value, confidence)
                                    VALUES (?, datetime('now'), ?, ?, ?, ?)
                                ''', (
                                    track_id, 
                                    'combined', 
                                    category,
                                    feature.get('feature', ''),
                                    feature.get('confidence', 0.0)
                                ))
            
            # Store top features if available
            if isinstance(top_features, list):
                for feature in top_features:
                    if isinstance(feature, dict):
                        self.cursor.execute('''
                            INSERT INTO track_features 
                            (track_id, timestamp, feature_type, category, feature_value, confidence)
                            VALUES (?, datetime('now'), ?, ?, ?, ?)
                        ''', (
                            track_id,
                            'top',
                            feature.get('type', 'unknown'),
                            feature.get('feature', ''),
                            feature.get('confidence', 0.0)
                        ))
            
            # Store style analysis if available
            if isinstance(style_analysis, dict):
                self.cursor.execute('''
                    INSERT INTO track_style 
                    (track_id, timestamp, dominant_colors, dominant_patterns, style_categories)
                    VALUES (?, datetime('now'), ?, ?, ?)
                ''', (
                    track_id,
                    json.dumps(style_analysis.get('dominant_colors', [])),
                    json.dumps(style_analysis.get('dominant_patterns', [])),
                    json.dumps(style_analysis.get('style_categories', []))
                ))
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            self.conn.rollback()
        except Exception as e:
            print(f"Error updating features: {e}")
            print(f"Feature matches: {feature_matches}")
            print(f"Style analysis: {style_analysis}")
            print(f"Top features: {top_features}")
            self.conn.rollback()
    
    def get_track_summary(self, track_id):
        """
        Get comprehensive summary for a track
        
        Args:
            track_id: Target tracking ID
            
        Returns:
            dict: Summary of track information including features and trajectory
        """
        try:
            # Get base information
            self.cursor.execute('''
                SELECT * FROM track_base WHERE track_id = ?
            ''', (track_id,))
            base_info = self.cursor.fetchone()
            
            if not base_info:
                return None
            
            # Get most common features using subquery
            self.cursor.execute('''
                WITH FeatureConfidence AS (
                    SELECT 
                        category,
                        feature_value,
                        AVG(confidence) as avg_confidence,
                        ROW_NUMBER() OVER (PARTITION BY category ORDER BY AVG(confidence) DESC) as rn
                    FROM track_features
                    WHERE track_id = ?
                    GROUP BY category, feature_value
                )
                SELECT category, feature_value, avg_confidence
                FROM FeatureConfidence
                WHERE rn = 1
                ORDER BY category
            ''', (track_id,))
            features = self.cursor.fetchall()
            
            # Process features into a dictionary
            feature_summary = {category: {'feature': feature_value, 'confidence': avg_confidence} for category, feature_value, avg_confidence in features}
            
            return {
                'base_info': base_info,
                'features': feature_summary
            }
        
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None
    
    def get_statistics(self):
        """
        Get overall tracking statistics
        
        Returns:
            dict: Statistical summary of all tracks
        """
        try:
            stats = {}
            
            # Total tracks and average duration
            self.cursor.execute('''
                SELECT COUNT(*) as total_tracks,
                       AVG(total_frames) as avg_frames,
                       MAX(total_frames) as max_frames
                FROM track_base
            ''')
            base_stats = self.cursor.fetchone()
            stats['total_tracks'] = base_stats[0]
            stats['avg_frames_per_track'] = base_stats[1]
            stats['max_frames'] = base_stats[2]
            
            # Feature statistics
            self.cursor.execute('''
                SELECT category, COUNT(DISTINCT feature_value) as unique_features
                FROM track_features
                GROUP BY category
            ''')
            feature_stats = self.cursor.fetchall()
            stats['feature_counts'] = {
                category: count for category, count in feature_stats
            }
            
            return stats
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None
    
    def batch_update_features(self, updates):
        """
        Perform batch update of features
        
        Args:
            updates: List of feature updates
        """
        try:
            self.cursor.executemany('''
                INSERT INTO track_features 
                (track_id, timestamp, feature_type, category, feature_value, confidence)
                VALUES (?, datetime('now'), ?, ?, ?, ?)
            ''', updates)
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            self.conn.rollback()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def recreate_tables(self):
        """Drop and recreate all tables - use with caution!"""
        try:
            self.cursor.executescript('''
                DROP TABLE IF EXISTS track_positions;
                DROP TABLE IF EXISTS track_features;
                DROP TABLE IF EXISTS track_style;
                DROP TABLE IF EXISTS track_base;
            ''')
            self.create_tables()
            
        except sqlite3.Error as e:
            print(f"Database error in recreate_tables: {e}")
            self.conn.rollback()
            raise
        except Exception as e:
            print(f"Unexpected error in recreate_tables: {e}")
            self.conn.rollback()
            raise