import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface ProfileData {
  experienceLevel: string;
  hardwareAccess: string[];
  contentDepthPreference: string;
  preferredTopics: string[];
  learningPace: string;
}

interface HardwareAlternative {
  original_hardware: string;
  alternatives: string[];
  simulation_option: string | null;
}

interface ProfileManagementProps {
  userId: string;
  backendUrl?: string;
  onClose?: () => void;
}

const ProfileManagement: React.FC<ProfileManagementProps> = ({
  userId,
  backendUrl = 'http://localhost:8000',
  onClose
}) => {
  const [profile, setProfile] = useState<ProfileData>({
    experienceLevel: 'beginner',
    hardwareAccess: [],
    contentDepthPreference: 'standard',
    preferredTopics: [],
    learningPace: 'standard'
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<'profile' | 'hardware' | 'preferences'>('profile');
  const [hardwareAlternatives, setHardwareAlternatives] = useState<HardwareAlternative[]>([]);

  useEffect(() => {
    fetchProfile();
  }, [userId]);

  const fetchProfile = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/profile?user_id=${userId}`);
      if (response.ok) {
        const data = await response.json();
        setProfile({
          experienceLevel: data.experience_level || 'beginner',
          hardwareAccess: data.hardware_access || [],
          contentDepthPreference: data.content_depth_preference || 'standard',
          preferredTopics: data.preferred_topics || [],
          learningPace: data.learning_pace || 'standard'
        });
      }
    } catch (error) {
      console.error('Error fetching profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveProfile = async () => {
    setSaving(true);
    try {
      const response = await fetch(`${backendUrl}/api/profile?user_id=${userId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experience_level: profile.experienceLevel,
          hardware_access: profile.hardwareAccess,
          content_depth_preference: profile.contentDepthPreference,
          preferred_topics: profile.preferredTopics,
          learning_pace: profile.learningPace
        })
      });

      if (response.ok) {
        // Show success message
        alert('Profile saved successfully!');
      }
    } catch (error) {
      console.error('Error saving profile:', error);
      alert('Error saving profile. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const fetchHardwareAlternatives = async (hardware: string) => {
    try {
      const response = await fetch(
        `${backendUrl}/api/profile/hardware-alternatives?hardware=${encodeURIComponent(hardware)}`
      );
      if (response.ok) {
        const data = await response.json();
        setHardwareAlternatives(prev => {
          const existing = prev.find(h => h.original_hardware === hardware);
          if (existing) return prev;
          return [...prev, data];
        });
      }
    } catch (error) {
      console.error('Error fetching alternatives:', error);
    }
  };

  const experienceLevels = [
    { id: 'beginner', label: 'Beginner', description: 'New to robotics and AI' },
    { id: 'intermediate', label: 'Intermediate', description: 'Some experience with robotics' },
    { id: 'advanced', label: 'Advanced', description: 'Professional or researcher' }
  ];

  const depthOptions = [
    { id: 'overview', label: 'Overview', description: 'High-level concepts, skip details' },
    { id: 'standard', label: 'Standard', description: 'Balanced coverage' },
    { id: 'detailed', label: 'Detailed', description: 'In-depth explanations and proofs' }
  ];

  const paceOptions = [
    { id: 'fast', label: 'Fast', description: 'Quick progression' },
    { id: 'standard', label: 'Standard', description: 'Normal pace' },
    { id: 'slow', label: 'Thorough', description: 'Take your time' }
  ];

  const hardwareOptions = [
    { id: 'nvidia_jetson', label: 'NVIDIA Jetson' },
    { id: 'raspberry_pi', label: 'Raspberry Pi' },
    { id: 'robot_arm', label: 'Robot Arm' },
    { id: 'humanoid_robot', label: 'Humanoid Robot' },
    { id: 'mobile_robot', label: 'Mobile Robot' },
    { id: 'lidar', label: 'LIDAR' },
    { id: 'depth_camera', label: 'Depth Camera' },
    { id: 'imu', label: 'IMU' }
  ];

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading profile...</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={styles.title}>Profile Settings</h2>
          {onClose && (
            <button onClick={onClose} style={styles.closeButton}>×</button>
          )}
        </div>

        {/* Tabs */}
        <div style={styles.tabs}>
          {[
            { id: 'profile', label: 'Experience' },
            { id: 'hardware', label: 'Hardware' },
            { id: 'preferences', label: 'Preferences' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                ...styles.tab,
                ...(activeTab === tab.id ? styles.tabActive : {})
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div style={styles.tabContent}>
          {activeTab === 'profile' && (
            <div>
              <h3 style={styles.sectionTitle}>Experience Level</h3>
              <p style={styles.sectionDescription}>
                This affects the depth of content shown in modules.
              </p>
              <div style={styles.optionsList}>
                {experienceLevels.map(level => (
                  <motion.div
                    key={level.id}
                    whileHover={{ scale: 1.01 }}
                    onClick={() => setProfile({ ...profile, experienceLevel: level.id })}
                    style={{
                      ...styles.optionItem,
                      ...(profile.experienceLevel === level.id ? styles.optionItemSelected : {})
                    }}
                  >
                    <div style={styles.optionRadio}>
                      {profile.experienceLevel === level.id && <span>●</span>}
                    </div>
                    <div>
                      <div style={styles.optionLabel}>{level.label}</div>
                      <div style={styles.optionDesc}>{level.description}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'hardware' && (
            <div>
              <h3 style={styles.sectionTitle}>Available Hardware</h3>
              <p style={styles.sectionDescription}>
                Select hardware you have access to. We'll suggest alternatives for what you don't have.
              </p>
              <div style={styles.checkboxList}>
                {hardwareOptions.map(hw => (
                  <div key={hw.id}>
                    <motion.div
                      whileHover={{ scale: 1.01 }}
                      onClick={() => {
                        const current = profile.hardwareAccess;
                        if (current.includes(hw.id)) {
                          setProfile({
                            ...profile,
                            hardwareAccess: current.filter(h => h !== hw.id)
                          });
                        } else {
                          setProfile({
                            ...profile,
                            hardwareAccess: [...current, hw.id]
                          });
                          fetchHardwareAlternatives(hw.id);
                        }
                      }}
                      style={{
                        ...styles.checkboxItem,
                        ...(profile.hardwareAccess.includes(hw.id) ? styles.checkboxItemSelected : {})
                      }}
                    >
                      <div style={styles.checkboxBox}>
                        {profile.hardwareAccess.includes(hw.id) && <span>✓</span>}
                      </div>
                      <span style={styles.checkboxLabel}>{hw.label}</span>
                    </motion.div>

                    {/* Show alternatives for unchecked hardware */}
                    {!profile.hardwareAccess.includes(hw.id) && (
                      <div style={styles.alternativesBox}>
                        {hardwareAlternatives
                          .filter(alt => alt.original_hardware === hw.id)
                          .map(alt => (
                            <div key={alt.original_hardware}>
                              <div style={styles.alternativesTitle}>Alternatives:</div>
                              <ul style={styles.alternativesList}>
                                {alt.alternatives.map((a, i) => (
                                  <li key={i}>{a}</li>
                                ))}
                              </ul>
                              {alt.simulation_option && (
                                <div style={styles.simulationOption}>
                                  💻 Simulation: {alt.simulation_option}
                                </div>
                              )}
                            </div>
                          ))}
                        {!hardwareAlternatives.find(alt => alt.original_hardware === hw.id) && (
                          <button
                            onClick={() => fetchHardwareAlternatives(hw.id)}
                            style={styles.showAlternativesBtn}
                          >
                            Show alternatives
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'preferences' && (
            <div>
              <h3 style={styles.sectionTitle}>Content Depth</h3>
              <p style={styles.sectionDescription}>
                How detailed should the content be?
              </p>
              <div style={styles.optionsList}>
                {depthOptions.map(option => (
                  <motion.div
                    key={option.id}
                    whileHover={{ scale: 1.01 }}
                    onClick={() => setProfile({ ...profile, contentDepthPreference: option.id })}
                    style={{
                      ...styles.optionItem,
                      ...(profile.contentDepthPreference === option.id ? styles.optionItemSelected : {})
                    }}
                  >
                    <div style={styles.optionRadio}>
                      {profile.contentDepthPreference === option.id && <span>●</span>}
                    </div>
                    <div>
                      <div style={styles.optionLabel}>{option.label}</div>
                      <div style={styles.optionDesc}>{option.description}</div>
                    </div>
                  </motion.div>
                ))}
              </div>

              <h3 style={{ ...styles.sectionTitle, marginTop: '30px' }}>Learning Pace</h3>
              <div style={styles.optionsList}>
                {paceOptions.map(option => (
                  <motion.div
                    key={option.id}
                    whileHover={{ scale: 1.01 }}
                    onClick={() => setProfile({ ...profile, learningPace: option.id })}
                    style={{
                      ...styles.optionItem,
                      ...(profile.learningPace === option.id ? styles.optionItemSelected : {})
                    }}
                  >
                    <div style={styles.optionRadio}>
                      {profile.learningPace === option.id && <span>●</span>}
                    </div>
                    <div>
                      <div style={styles.optionLabel}>{option.label}</div>
                      <div style={styles.optionDesc}>{option.description}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Save Button */}
        <div style={styles.footer}>
          <button
            onClick={saveProfile}
            disabled={saving}
            style={styles.saveButton}
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.85)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10000,
    padding: '20px'
  },
  modal: {
    background: 'linear-gradient(145deg, rgba(30, 30, 30, 0.98), rgba(20, 20, 20, 0.98))',
    borderRadius: '16px',
    maxWidth: '600px',
    width: '100%',
    maxHeight: '85vh',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid rgba(136, 136, 136, 0.2)'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '24px',
    borderBottom: '1px solid rgba(136, 136, 136, 0.2)'
  },
  title: {
    fontFamily: 'Sora, sans-serif',
    color: '#fff',
    fontSize: '1.5rem',
    fontWeight: 700,
    margin: 0
  },
  closeButton: {
    background: 'transparent',
    border: 'none',
    color: '#888',
    fontSize: '24px',
    cursor: 'pointer',
    padding: '4px 8px'
  },
  tabs: {
    display: 'flex',
    borderBottom: '1px solid rgba(136, 136, 136, 0.2)',
    padding: '0 24px'
  },
  tab: {
    background: 'transparent',
    border: 'none',
    color: '#888',
    padding: '16px 24px',
    cursor: 'pointer',
    fontFamily: 'Sora, sans-serif',
    fontSize: '0.95rem',
    borderBottom: '2px solid transparent',
    marginBottom: '-1px'
  },
  tabActive: {
    color: '#fff',
    borderBottomColor: '#888'
  },
  tabContent: {
    padding: '24px',
    overflow: 'auto',
    flex: 1
  },
  sectionTitle: {
    fontFamily: 'Sora, sans-serif',
    color: '#fff',
    fontSize: '1.1rem',
    fontWeight: 600,
    marginBottom: '8px'
  },
  sectionDescription: {
    fontFamily: 'Inter, sans-serif',
    color: '#888',
    fontSize: '0.9rem',
    marginBottom: '20px'
  },
  optionsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px'
  },
  optionItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '16px',
    background: 'rgba(40, 40, 40, 0.5)',
    borderRadius: '10px',
    cursor: 'pointer',
    border: '2px solid transparent',
    transition: 'all 0.2s ease'
  },
  optionItemSelected: {
    borderColor: '#888',
    background: 'rgba(136, 136, 136, 0.1)'
  },
  optionRadio: {
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    border: '2px solid #666',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#888'
  },
  optionLabel: {
    fontFamily: 'Inter, sans-serif',
    color: '#d0d0d0',
    fontSize: '1rem',
    fontWeight: 500
  },
  optionDesc: {
    fontFamily: 'Inter, sans-serif',
    color: '#888',
    fontSize: '0.85rem',
    marginTop: '2px'
  },
  checkboxList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  checkboxItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px 16px',
    background: 'rgba(40, 40, 40, 0.5)',
    borderRadius: '8px',
    cursor: 'pointer',
    border: '2px solid transparent'
  },
  checkboxItemSelected: {
    borderColor: '#888',
    background: 'rgba(136, 136, 136, 0.1)'
  },
  checkboxBox: {
    width: '20px',
    height: '20px',
    borderRadius: '4px',
    border: '2px solid #666',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#888',
    fontSize: '12px',
    fontWeight: 'bold'
  },
  checkboxLabel: {
    fontFamily: 'Inter, sans-serif',
    color: '#d0d0d0',
    fontSize: '0.95rem'
  },
  alternativesBox: {
    marginLeft: '44px',
    marginTop: '8px',
    marginBottom: '16px',
    padding: '12px',
    background: 'rgba(136, 136, 136, 0.05)',
    borderRadius: '8px',
    borderLeft: '3px solid #555'
  },
  alternativesTitle: {
    fontFamily: 'Sora, sans-serif',
    color: '#888',
    fontSize: '0.8rem',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginBottom: '8px'
  },
  alternativesList: {
    fontFamily: 'Inter, sans-serif',
    color: '#a0a0a0',
    fontSize: '0.9rem',
    margin: '0 0 0 16px',
    padding: 0
  },
  simulationOption: {
    fontFamily: 'Inter, sans-serif',
    color: '#888',
    fontSize: '0.85rem',
    marginTop: '8px',
    fontStyle: 'italic'
  },
  showAlternativesBtn: {
    background: 'transparent',
    border: '1px solid #555',
    color: '#888',
    padding: '6px 12px',
    borderRadius: '4px',
    cursor: 'pointer',
    fontFamily: 'Inter, sans-serif',
    fontSize: '0.8rem'
  },
  footer: {
    padding: '20px 24px',
    borderTop: '1px solid rgba(136, 136, 136, 0.2)',
    display: 'flex',
    justifyContent: 'flex-end'
  },
  saveButton: {
    background: 'linear-gradient(135deg, #888888, #e0e0e0)',
    border: 'none',
    color: '#000',
    padding: '12px 32px',
    borderRadius: '8px',
    cursor: 'pointer',
    fontFamily: 'Sora, sans-serif',
    fontSize: '1rem',
    fontWeight: 600
  },
  loading: {
    color: '#888',
    fontFamily: 'Inter, sans-serif',
    fontSize: '1rem'
  }
};

export default ProfileManagement;
