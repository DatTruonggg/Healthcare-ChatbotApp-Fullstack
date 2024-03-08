import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { createDrawerNavigator } from '@react-navigation/drawer';
import { UserContext } from './components/UserContext';
import React from 'react';
import HomeChatScreen from './HomeChatScreen';
import FriendsScreen from "./FriendsScreen";
import ChatsScreen from "./ChatsScreen";
import ChatMessagesScreen from "./ChatMessagesScreen";
import { appColors } from '../../constants/appColors'; // Import màu từ file appColors
import HomeScreen from '../home/HomeScreen'
import DrawerCustom from '../../components/DrawerCustom';
import AIChatScreen from './AiChat/AIChatScreen'
const Stack = createNativeStackNavigator();
const Drawer = createDrawerNavigator();

function MainStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="HomeChatScreen" component={HomeChatScreen} />
      <Stack.Screen name="Friends" component={FriendsScreen} />
      <Stack.Screen name="Chats" component={ChatsScreen} />
      <Stack.Screen name="Messages" component={ChatMessagesScreen} />
      <Stack.Screen name='AIChatScreen' component={AIChatScreen} options={{ headerShown: false }}/>
    </Stack.Navigator>
  );
}

export default function ChatApp() {
  return (
    <UserContext>
        <Drawer.Navigator
          screenOptions={{
            drawerBackgroundColor: appColors.drawerBackground}}
            drawerContent={props => <DrawerCustom {...props} />}>
          <Drawer.Screen name="Chat" component={MainStack} options={{ headerShown: false }}/>          
        </Drawer.Navigator>
    </UserContext>
  );
}
