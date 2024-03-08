import { View, Text, Dimensions } from 'react-native';
import React from 'react';
import { EventModels } from '../screens/models/EventModels'
import { Car, Location } from 'iconsax-react-native';
import { ImageBackground } from 'react-native';
import {
  SpaceComponent,
  CardComponent,
  TextComponent,
  RowComponent,
} from '.';
import { appInfo } from "../constants/appInfos";


interface Props {
  item: EventModels;
  type: 'card' | 'list';
}

const EventItem = (props: Props) => {
  const { item, type } = props;
  return (
    <CardComponent styles={{ width: appInfo.sizes.WIDTH * 0.7 }} onPress={() => { }} children={undefined}></CardComponent>

  );

};
export default EventItem