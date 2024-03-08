import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, FlatList, ScrollView } from 'react-native';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import {
  faCloud, faLink, faMicrophone, faPaperPlane, faSun,
  faThunderstorm, faArrowLeft, faCircleInfo
} from '@fortawesome/free-solid-svg-icons';
import { LinearGradient } from 'react-native-linear-gradient';
import { useNavigation } from "@react-navigation/native";
import { basic, center, form } from '../../../styles/style';
import { Image } from 'react-native';
import Markdown from 'react-native-markdown-display';
import axios from 'axios'; 


const AIChatScreen = () => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const navigation = useNavigation();

  const handleBackButton = () => {
    navigation.goBack();
  };

  //  const handleSend = async () => {
  //   if (newMessage.trim() === '') {
  //     return;
  //   }
  //   try {
  //     const response = await fetch(`http://localhost:8080/generate-response`, {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({
  //         prompt: newMessage,
  //       }),
  //     });

  //     if (response.ok) {
  //       const result = await response.json();
  //       setMessages((prevMessages) => [
  //         ...prevMessages,
  //         { id: String(prevMessages.length + 1), text: newMessage, sender: 'User' },
  //         { id: String(prevMessages.length + 2), text: result.response, sender: 'Friend' },
  //       ]);
  //       setNewMessage('');
  //     } else {
  //       console.error('Error getting response:', response.statusText);
  //     }
  //   } catch (error) {
  //     console.error('Unhandled promise rejection:', error);
  //   }
  // };

  const handleSend = async () => {
    if (newMessage.trim() === '') {
      return;
    }
    try {
      const response = await axios.post('http://10.0.2.2:8080/generate-response', {
        prompt: newMessage,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      if (response.status === 200) {
        const result = response.data;
        setMessages((prevMessages) => [
          ...prevMessages,
          { id: String(prevMessages.length + 1), text: newMessage, sender: 'User' },
          { id: String(prevMessages.length + 2), text: result.response, sender: 'Friend' },
        ]);
        setNewMessage('');
      } else {
        console.error('Error getting response:', response.statusText);
      }
    } catch (error) {
      console.error('Unhandled promise rejection:', error);
    }
  };
  const handleLink = () => {
    console.log('Handling link:', messages);
  };

  const handleVoice = () => {
    console.log('Handling Voice:', messages);
  };

  return (
    <LinearGradient colors={['#ffffff', '#F8FAFE', '#F0F4FC', '#DCE6F9', '#DBE6F9', '#E6EEFB', '#DDE7F9', '#E0E9FA', '#9EB8D9']} start={{ x: 0, y: 0 }} end={{ x: 0, y: 1 }} style={{...basic.gradientBox, marginTop:20}}>
      
      <View style={form.topContainer}>
        <TouchableOpacity onPress={handleBackButton} style={{ marginLeft: 8 }}>
          <FontAwesomeIcon icon={faArrowLeft} size={25} style={form.iconBack} color='#646464' />
        </TouchableOpacity>
        <FontAwesomeIcon icon={faCircleInfo} size={25} style={form.iconInf} color='#646464' />
      </View>
      <ScrollView>
        <View>
          <Image source={require('D:/DatTruong/All/2025/Capstone_final/Chatbot/src/assets/images/another-logo.png')}
          style={{
            alignSelf:'center',
            resizeMode: 'contain', 
            width: 100, 
            height: 150, 
            marginTop: -30
          }}/>
        </View>
        {messages.length === 0 && (
          <View style={form.header}>
            <Text style={form.title}>Xin chào, hãy hỏi điều gì đó...</Text>
          </View>
        )}
        {messages.length === 0 && (
          <View>
            <FontAwesomeIcon style={center.icon} icon={faSun} size={50} color='#BF95E4' />
            <TouchableOpacity onPress={() => handleSend('Tôi gặp một vài triệu chứng')} style={center.containerBox1}>
              <Text style={center.text}>Tôi gặp một vài triệu chứng</Text>
            </TouchableOpacity>
          </View>
        )}

        {messages.length === 0 && (
          <View>
            <TouchableOpacity onPress={() => handleSend("Tôi có thể hỏi triệu chứng của bệnh...")} style={center.containerBox2}>
              <Text style={center.text}>Tôi có thể hỏi triệu chứng của bệnh...</Text>
            </TouchableOpacity>
          </View>
        )}

        {messages.length === 0 && (
          <View>
            <FontAwesomeIcon style={center.icon1} icon={faCloud} size={50} color='#55B9AC' />
            <TouchableOpacity onPress={() => handleSend("Tôi muốn hỏi là...")} style={center.containerBox3}>
              <Text style={center.text}>Tôi muốn hỏi là...</Text>
            </TouchableOpacity>
          </View>
        )}
        {messages.length === 0 && (
          <View>
            <FontAwesomeIcon style={center.icon2} icon={faThunderstorm} size={50} color='#FFB224' />
            <TouchableOpacity onPress={() => handleSend("Có ai ở đó không?")} style={center.containerBox4}>
              <Text style={center.text}>Có ai ở đó không?</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>

      <FlatList
        data={messages}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={item.sender === 'User' ? form.messageContainer : form.friendMessageContainer}>
            <Markdown style={[form.message, { color: item.sender === 'User' ? 'white' : '#000212' }]}>{item.text}</Markdown>
          </View>
        )}
      />

      <View style={form.iconsContainerLink}>
        <TouchableOpacity onPress={handleLink} style={form.iconLink}>
          <FontAwesomeIcon icon={faLink} size={25} color="#3C61DD" />
        </TouchableOpacity>
        <TouchableOpacity onPress={handleVoice} style={form.iconVoice}>
          <FontAwesomeIcon icon={faMicrophone} size={25} color='#3C61DD' />
        </TouchableOpacity>
        <TextInput
          style={form.input}
          placeholder="Type a message..."
          value={newMessage}
          onChangeText={(text) => setNewMessage(text)}
        />
        <TouchableOpacity onPress={handleSend} style={form.iconSend}>
          <FontAwesomeIcon icon={faPaperPlane} size={25} color="#3C61DD" />
        </TouchableOpacity>
      </View>
    </LinearGradient>
  );
};

export default AIChatScreen;
