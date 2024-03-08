import { StyleSheet, Text, View, TouchableOpacity } from "react-native";
import React, { useLayoutEffect, useContext, useEffect, useState } from "react";
import { useNavigation } from "@react-navigation/native";
import Ionicons from "react-native-vector-icons/Ionicons"; // Import Ionicons from react-native-vector-icons
import MaterialIcons from "react-native-vector-icons/MaterialIcons"; // Import MaterialIcons from react-native-vector-icons
import { UserType } from "./components/UserContext";
import AsyncStorage from "@react-native-async-storage/async-storage";
import axios from "axios";
import User from "./components/User";
import {
  ArrowLeft,
} from 'iconsax-react-native';
import { FontAwesomeIcon } from "@fortawesome/react-native-fontawesome";
import { faUserFriends, faMessage } from "@fortawesome/free-solid-svg-icons";
import { jwtDecode } from "jwt-decode";

const HomeChatScreen = () => {
  const navigation = useNavigation();
  const { userId, setUserId } = useContext(UserType);
  const [users, setUsers] = useState([]);
  useLayoutEffect(() => {
    navigation.setOptions({
      headerTitle:'',
      headerLeft: () => (
        <View style={{ flexDirection: "row", alignItems: "center", marginLeft: 10 }}>
          <ArrowLeft onPress={() => navigation.goBack()} size={20} color="black" marginLeft={-15} />
          <Text style={{ fontSize: 25, fontWeight: "bold", marginLeft: 10 }}>Chat</Text>
        </View>
      ),
      headerRight: () => (
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          <FontAwesomeIcon onPress={() => navigation.navigate("Chats")} icon={faMessage} size={24} color="black" />
          <FontAwesomeIcon
            icon={faUserFriends} 
            onPress={() => navigation.navigate("Friends")}
            size={24}
            color="black"
          />
        </View>
      ),
    });
  }, []);

  useEffect(() => {
    const fetchUsers = async () => {
      const token = await AsyncStorage.getItem("authToken");
      const decodedToken = jwtDecode(token);
      const userId = decodedToken.userId;
      setUserId(userId);

      axios
        .get(`http://192.168.199.149:8080/user/${userId}`)
        .then((response) => {
          setUsers(response.data);
        })
        .catch((error) => {
          console.log("error retrieving users", error);
        });
    };

    fetchUsers();
  }, []);

  console.log("users", users);
  return (
    <View>
      <TouchableOpacity 
        onPress={() => navigation.navigate("AIChatScreen")}
        style={{
          marginTop:10,
          backgroundColor: '#46cdfb',
          padding: 10,
          borderRadius: 8, // Điều này sẽ tạo góc bo tròn cho nút
          shadowColor: '#000', // Màu của shadow
          shadowOffset: { width: 2, height: 10 }, // Độ dịch chuyển của shadow
          shadowOpacity: 0.4, // Độ mờ của shadow
          shadowRadius: 2, // Bán kính của shadow
          elevation: 5, // Độ cao tạo ra shadow (chỉ áp dụng cho Android)
        }}
      >
        <Text style={{ fontSize: 16, fontWeight: 'bold', color: '#000' }}>Chat AI</Text>
      </TouchableOpacity>
      <View style={{ padding: 10 }}>
        {users.map((item, index) => (
          <User key={index} item={item} />
        ))}
      </View>

    </View>
  );
};

export default HomeChatScreen;

const styles = StyleSheet.create({});