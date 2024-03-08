import { View, Text, FlatList } from 'react-native';
import React, { ReactNode } from 'react';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { RowComponent, SpaceComponent, TextComponent } from '.';
import { globalStyles } from '../styles/globalStyles';
import { appColors } from '../constants/appColors';
import FontAwesome from 'react-native-vector-icons/FontAwesome';
import { ChefFork } from '../assets/svgs';
import {
  Book1,
  Star,
  StarSlash
} from 'iconsax-react-native';

interface Props {
  isColor?: boolean;
}

interface Category {
  key: string;
  title: string;
  icon: ReactNode;
  iconColor: string;
}

const CategoriesList = (props: Props) => {
  const { isColor } = props;

  const categories: Category[] = [
    {
      key: '1',
      icon: (
        <Book1
          style={{ margin: -5 }}
          size={22}
          color={isColor ? appColors.white : '#EE544A'}
        />
      ),
      iconColor: '#EE544A',
      title: 'Tin tức',
    },
    {
      key: '2',
      icon: (
        <Star
          style={{ margin: -5 }}
          size={22}
          color={isColor ? appColors.white : '#F59762'}
        />
      ),
      iconColor: '#F59762',
      title: 'Tips',
    },
    {
      key: '3',
      icon: <ChefFork color={isColor ? appColors.white : '#29D697'} />,
      iconColor: '#29D697',
      title: 'Thực phẩm',
    },
    {
      key: '4',
      icon: (
        <StarSlash
          size={22}
          color={isColor ? appColors.white : '#46CDFB'}
        />
      ),
      iconColor: '#46CDFB',
      title: 'Khác',
    },
  ];

  const renderTagCategory = (item: Category) => {
    return (
      <RowComponent
        onPress={() => { }}
        styles={[
          globalStyles.tag,
          {
            backgroundColor: isColor ? item.iconColor : appColors.white,
          },
        ]}>
        {item.icon}
        <SpaceComponent width={8} />
        <TextComponent
          text={item.title}
          color={isColor ? appColors.white : appColors.gray}
        />
      </RowComponent>
    );
  };

  return (
    <FlatList
      style={{ paddingHorizontal: 16 }}
      showsHorizontalScrollIndicator={false}
      horizontal
      data={categories}
      renderItem={({ item }) => renderTagCategory(item)}
    />
  );
};

export default CategoriesList;